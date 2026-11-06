"""Cheap-fix arms for the language-mixing experiment. Everything here uses practice data only (runs/langmix/practice).

    python langmix/baselines.py list      # the eval.py commands for every baseline arm
    python langmix/baselines.py ban       # build + sanity-check the CJK token list (CPU only)
    python langmix/baselines.py steer     # difference-of-means steering vector (one GPU lock, ~1-2 min)

Arms (all scored by langmix/eval.py; see `list`). Framing: Chinese leaking into de/ru/th answers.
- instr     "Reply in the same language as the question." appended to the user turn on its own line (DeepSeek's usage
            notes for R1 models advise against system prompts; --instr-role system is the optional variant).
- ban       logits mask over every vocab token that can emit a CJK character (common.cjk_token_ids: whole-character
            tokens plus UTF-8 byte fragments with a lead byte 0xE3-0xED). Equivalent to bad_words_ids with one
            entry per token. Fails Chinese retention by design.
- steer     (optional) difference-of-means steering toward the prompt's language. Direction at layer L:
              v = mean residual(non-Chinese completions) - mean residual(Chinese-prompt completions)
            from teacher-forcing the base model's own practice completions: 20 `clean` + 20 `mix` de/ru/th ones
            (for `mix`, only the tokens before the first CJK character) and 40 `zh` ones, at most 512 completion
            tokens each, prompt tokens excluded. Layer L is chosen without looking at any outcome: the layer in the
            middle third [n/3, 2n/3] where the per-completion means projected on v separate best (difference of
            class means / pooled std). The hook adds alpha * v to the residual stream after layer L at every
            position: mode cond (default) uses +alpha for non-Chinese prompts and -alpha for Chinese prompts, i.e.
            toward the prompt's language, with the prompt language read from the prompt text (Han share > 0.2);
            mode cond0 leaves Chinese prompts alone; mode all adds +alpha everywhere. alpha = 1 adds one full mean
            difference. alpha is picked from {0.25, 0.5, 1} on practice data only (`list` prints the commands): the
            largest alpha whose practice_zh zh_reply and practice_mix accuracy stay within the pass-bar margins of
            alpha 0, else the one with the lowest practice_mix mixing among those that do.
            Weight-space equivalent: none that is clean. Qwen2's o_proj, down_proj and lm_head have no bias, so a
            constant added to the residual stream at layer L > 0 cannot be written into the weights exactly, and
            the conditional modes depend on the prompt language, which no static weight can read without becoming
            a different intervention. The hook arm is the steering baseline; it is an inference-time change, like
            the ban, that ollama / llama.cpp users cannot apply.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.RUNS / "baselines"
N_PER_SIDE = 40
MAX_COMP = 512


def cmd_list(a):
    py = "PYTHONIOENCODING=utf-8 .venv/Scripts/python langmix/eval.py"
    steer = (OUT / "steer.pt").relative_to(C.ROOT).as_posix()
    print(f"{py} --arm base")
    print(f"{py} --arm instr --instr same")
    print(f"{py} --arm ban --ban-cjk")
    print("# steering (optional): tune alpha on practice data only, then one full arm with the chosen alpha")
    print(f"{py} --arm steer_tune_a0 --suites practice_mix,practice_zh --limit 48")
    for al in ("0.25", "0.5", "1"):
        print(f"{py} --arm steer_tune_a{al} --steer {steer} --steer-alpha {al} --suites practice_mix,practice_zh --limit 48")
    print(f"{py} --arm steer --steer {steer} --steer-alpha <chosen>")


def cmd_ban(a):
    tok = C.load_tok(a.model)
    ids = C.cjk_token_ids(tok)
    sample = [tok.decode([i]) for i in ids[:: max(1, len(ids) // 25)]]
    print(f"{len(ids):,} of {len(tok):,} tokens banned; sample: {sample}")
    for s in ["首先", "我们", "因此", "答案", "こんにちは", "한국어"]:
        enc = tok(s, add_special_tokens=False).input_ids
        print(f"  {s!r} -> {enc} all banned: {all(i in set(ids) for i in enc)}")
    for s in ["Therefore, the answer is 42.", "x^2 + y^2 = 1", "café — naïve", "→ ≤ ≥ ∞ π"]:
        enc = tok(s, add_special_tokens=False).input_ids
        print(f"  {s!r}: banned tokens {sum(i in set(ids) for i in enc)}/{len(enc)}")


@torch.no_grad()
def cmd_steer(a):
    import mine_practice as MP
    P = MP.load_practice(a.practice)
    tok = C.load_tok(a.model)
    model = C.load_model(a.model)
    n_layers = model.config.num_hidden_layers

    def comp_ids(item, english_only):
        text = item["completion"]
        if english_only:
            m = C.CJK.search(text)
            text = text[: m.start()] if m else text
        ids = tok(text, add_special_tokens=False).input_ids[:MAX_COMP]
        return ids

    sides = dict(en=[(it, True) for it in P["clean"][:N_PER_SIDE // 2] + P["mix"][:N_PER_SIDE // 2]],
                 zh=[(it, False) for it in P["zh"][:N_PER_SIDE]])
    means = {}  # side -> [n_texts, n_layers + 1, d] per-completion mean residual (hidden_states index L+1 = after L)
    with C.gpu("langmix/baselines.py steer"):
        torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC)
        model.cuda()
        for side, items in sides.items():
            rows = []
            for it, eng in items:
                p = tok(C.chat_prompt(tok, C.user_text(it)), add_special_tokens=False).input_ids
                c = comp_ids(it, eng)
                if len(c) < 8:
                    continue
                hs = model(input_ids=torch.tensor([p + c]).cuda(), output_hidden_states=True).hidden_states
                rows.append(torch.stack([h[0, len(p):].float().mean(0) for h in hs]).cpu())
            means[side] = torch.stack(rows)
        model.cpu()
    en, zh = means["en"], means["zh"]
    diff = en.mean(0) - zh.mean(0)                     # [n_layers + 1, d]
    sep = []
    for L in range(n_layers + 1):
        u = diff[L] / diff[L].norm()
        pe, pz = en[:, L] @ u, zh[:, L] @ u
        pooled = ((pe.var() + pz.var()) / 2).sqrt()
        sep.append(float((pe.mean() - pz.mean()) / pooled))
    lo, hi = n_layers // 3, (2 * n_layers) // 3  # middle third: the usual steering range; late layers separate
    # languages trivially (they are close to the output tokens), so they are excluded by rule, not by outcome
    best = max(range(lo, hi + 1), key=lambda L: sep[L + 1])  # hidden_states[L + 1] = output of decoder layer L
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(layer=best, vec=diff[best + 1].clone(), sep=sep, n_en=len(en), n_zh=len(zh),
                    all_layers=diff.clone(), note="v = mean resid(English completions) - mean resid(Chinese completions)"
                    " after decoder layer `layer`; add alpha * v"), OUT / "steer.pt")
    info = dict(layer=best, norm=round(float(diff[best + 1].norm()), 3),
                resid_norm=round(float(en[:, best + 1].norm(dim=-1).mean()), 3), n_en=len(en), n_zh=len(zh),
                separation_by_layer=[round(s, 2) for s in sep])
    (OUT / "steer.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "ban", "steer"])
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--practice", default=str(C.RUNS / "practice"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    globals()["OUT"] = Path(a.out)
    dict(list=cmd_list, ban=cmd_ban, steer=cmd_steer)[a.cmd](a)


if __name__ == "__main__":
    main()
