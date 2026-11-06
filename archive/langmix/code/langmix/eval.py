"""Score one arm of the language-mixing experiment on the fixed suites (langmix/data.py).

Framing: Chinese leaking into German, Russian and Thai answers (MGSM, native-language question only, no instruction).

An arm = model + optional weight edit + optional instruction + optional decoding mods:
    python langmix/eval.py --arm base
    python langmix/eval.py --arm instr      --instr same          # "Reply in the same language as the question."
    python langmix/eval.py --arm ban        --ban-cjk             # CJK-token ban (logits mask)
    python langmix/eval.py --arm steer_a05  --steer runs/langmix/baselines/steer.pt --steer-alpha 0.5 --steer-mode cond0
    python langmix/eval.py --arm s1_agent   --edit runs/langmix/sessions/s1/submitted_edit.safetensors
    python langmix/eval.py --arm s1_rand0_same --edit runs/langmix/sessions/s1/random0_same_edit.safetensors
Options: --suites (default mgsm_de,mgsm_ru,mgsm_th,mgsm_en,mgsm_zh,ifeval,chat) --limit N (first N items per suite)
         --batch 24 --max-new 2048 --model <hub id or local dir> --target-lang xx --no-reuse-check --out-root DIR
Extra suites for tuning baselines on practice data only: practice_mix, practice_clean, practice_zh, practice_en
(the sets mined by mine_practice.py). Older suites: math500, gsm_sym.

Generation settings are check.py's sampling (T=0.6, top_p=0.95, top-k off, 2,048 new tokens), no system prompt,
chunks of 24 with seed 1000 + index of the chunk's first item. MGSM and practice prompts are the question only
(check_multi.py's format); math500/gsm_sym add check.py's suffix. For the pure base arm, rows from the viability checks
with identical prompts and settings are imported and rescored instead of regenerated: check_multi.py's first 16
de/ru/th problems (48 rows) and check.py's MATH-500 rows.

`chat` is not generated per arm: the base model's sampled 256-token answers to the 150 chat prompts are cached once
(runs/langmix/data/chat_ref_<model>.json) and the arm is scored by teacher forcing them: mean per-token
KL(base || arm) of the next-token distributions over the answer tokens, each side under its own prompt (so an
instruction arm is charged for what its instruction changes) and decoding mods (the ban's -inf logits are floored
at log 1e-10 so the KL stays finite). Also top-1 agreement and the probability mass on CJK tokens.

Resumable: every response is appended to runs/langmix/eval/<arm>/<suite>.jsonl, finished ids are skipped.
Each chunk takes the GPU lock separately (~2.5-4 min for 24 x 2,048 tokens). summary.json is rewritten per suite.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import data as D  # noqa: E402
import metrics as M  # noqa: E402

EVAL = C.RUNS / "eval"
KL_TOKENS = 256
KL_FLOOR = -23.0  # log(1e-10)
ALL = ("mgsm_de", "mgsm_ru", "mgsm_th", "mgsm_en", "mgsm_zh", "ifeval", "chat")


# ---------------------------------------------------------------- arm

class Arm:
    """Everything that defines an arm, applied consistently to generation and teacher forcing."""

    def __init__(self, a, model, tok):
        self.a, self.model, self.tok = a, model, tok
        self.instr = a.instr_text or (C.INSTRUCTIONS[a.instr] if a.instr else None)
        self.edit_orig = {}
        if a.edit:
            self.edit = C.load_edit(a.edit)
            self.edit_orig = C.apply_edit(model, self.edit)
        self.ban = None
        if a.ban_cjk:
            ids = C.cjk_token_ids(tok)
            self.ban = C.BanProcessor(ids, model.config.vocab_size)
            self.n_banned = len(ids)
        self.steer = None
        if a.steer:
            s = torch.load(a.steer, map_location="cpu")
            self.steer = dict(layer=int(s["layer"]), vec=s["vec"].float(), mode=a.steer_mode, alpha=a.steer_alpha)

    @property
    def is_base(self):
        a = self.a
        return not (a.edit or self.instr or a.ban_cjk or a.steer)

    def prompt(self, item):
        return C.chat_prompt(self.tok, C.user_text(item), self.instr, self.a.instr_role)

    def alphas(self, items):
        """Per-row steering coefficient. The prompt language is read from the user text (Han share > 0.2 = Chinese),
        as a serving stack would, not from the suite label."""
        s = self.steer
        out = []
        for it in items:
            zh = M.han_frac(it["question"]) > 0.2
            out.append(s["alpha"] if not zh else {"all": s["alpha"], "cond": -s["alpha"], "cond0": 0.0}[s["mode"]])
        return out

    def gen_kwargs(self, items):
        kw = {}
        if self.ban is not None:
            kw["processors"] = [self.ban]
        if self.steer is not None:
            kw["steer"] = (self.steer["layer"], self.steer["vec"], self.alphas(items))
        return kw

    def config(self):
        a = self.a
        return dict(arm=a.arm, model=a.model, edit=a.edit, instruction=self.instr, instr_role=a.instr_role if self.instr else None,
                    ban_cjk=a.ban_cjk, n_banned=getattr(self, "n_banned", None),
                    steer=a.steer, steer_alpha=a.steer_alpha if a.steer else None,
                    steer_mode=a.steer_mode if a.steer else None, steer_layer=self.steer["layer"] if self.steer else None,
                    edit_tensors=sorted(self.edit_orig), max_new=a.max_new, batch=a.batch, target_lang=a.target_lang)


# ---------------------------------------------------------------- suites

def suite_items(name):
    if name.startswith("practice_"):
        import mine_practice as MP
        return MP.load_practice()[name.split("_", 1)[1]]
    return D.load(name)


TARGET = None  # --target-lang; None = each item's prompt language


def score_row(item, text, n_tok, trunc):
    r = dict(id=item["id"], length=n_tok, truncated=trunc)
    r.update(M.language_fields(text, item.get("lang", "en"), TARGET))
    if item["kind"] in ("math", "mgsm"):
        if item["scorer"] == "math":
            pred = C.last_boxed(text)
            from check import norm
            r.update(pred=pred, gold=item["gold"], correct=M.math_equal(pred, item["gold"]),
                     correct_strict=pred is not None and norm(pred) == norm(item["gold"]), boxed=pred is not None)
        else:
            pred, boxed = M.numeric_pred(text)
            r.update(pred=pred, gold=item["gold"], correct=M.numeric_equal(pred, item["gold"]), boxed=boxed)
    elif item["kind"] == "ifeval":
        from ifeval_check import process  # the response scored is the text after </think> ("" if none)
        _, answer = M.split_think(text)
        res = process(item["doc"], [answer or ""])
        r.update(strict=bool(res["prompt_level_strict_acc"]), loose=bool(res["prompt_level_loose_acc"]),
                 inst_strict=list(map(bool, res["inst_level_strict_acc"])),
                 inst_loose=list(map(bool, res["inst_level_loose_acc"])))
    r["completion"] = text
    return r


def import_check_rows(a, name, out, items):
    """Base arm only: reuse a viability check's generations (identical prompt and settings), rescored.
    math500 <- runs/langmix/check/<model>/math500.jsonl; mgsm_xx <- runs/langmix/check_multi/<model>/mgsm.jsonl
    (ids "<lang>-<index>")."""
    tag = C.model_tag(a.model)
    if name == "math500":
        src, key = C.RUNS / "check" / tag / "math500.jsonl", lambda r: r["id"]
    elif name.startswith("mgsm_"):
        src, key = C.RUNS / "check_multi" / tag / "mgsm.jsonl", lambda r: f"mgsm-{r['id']}"
    else:
        return
    have = {r["id"] for r in C.read_jsonl(out)}
    by_id = {it["id"]: it for it in items}
    rows = [score_row(by_id[key(r)], r["completion"], r["length"], r["truncated"]) | dict(source=src.parent.parent.name)
            for r in C.read_jsonl(src) if key(r) in by_id and key(r) not in have]
    C.append_jsonl(out, rows)
    if rows:
        print(f"  imported {len(rows)} rows from {src}", flush=True)


def run_gen_suite(arm, name, odir):
    a = arm.a
    items = suite_items(name)[: a.limit]
    out = odir / f"{name}.jsonl"
    if arm.is_base and not a.no_reuse_check and a.max_new == C.MAX_NEW and a.target_lang is None:
        import_check_rows(a, name, out, items)
    done = {r["id"] for r in C.read_jsonl(out)}
    for start in range(0, len(items), a.batch):
        chunk = [it for it in items[start:start + a.batch] if it["id"] not in done]
        if not chunk:
            continue
        gens, secs = C.generate_chunk(arm.model, arm.tok, [arm.prompt(it) for it in chunk], C.SEED0 + start,
                                      a.max_new, f"langmix/eval.py {a.arm} {name} {start}", **arm.gen_kwargs(chunk))
        C.append_jsonl(out, [score_row(it, *g) | dict(gpu_s=round(secs / len(chunk), 2))
                             for it, g in zip(chunk, gens)])
        print(f"  {name} {start + len(chunk)}/{len(items)}: {len(chunk)} in {secs:.0f}s GPU", flush=True)
    rows = C.read_jsonl(out)
    ids = {it["id"] for it in items}
    return [r for r in rows if r["id"] in ids]


def summarize_gen(name, rows):
    n = len(rows) or 1
    mean = lambda k: round(sum(float(r.get(k) or 0) for r in rows) / n, 4)  # noqa: E731

    def mean_def(k):  # mean over rows where the field is defined
        v = [r[k] for r in rows if r.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None
    s = dict(n=len(rows), trunc=mean("truncated"), mean_len=round(sum(r["length"] for r in rows) / n),
             has_answer=mean("has_answer"), gpu_min=round(sum(r.get("gpu_s", 0) for r in rows) / 60, 1))
    if name == "ifeval":
        ni = sum(len(r["inst_strict"]) for r in rows) or 1
        s.update(prompt_strict=mean("strict"), prompt_loose=mean("loose"),
                 inst_strict=round(sum(sum(r["inst_strict"]) for r in rows) / ni, 4),
                 inst_loose=round(sum(sum(r["inst_loose"]) for r in rows) / ni, 4),
                 mix=mean("mix"), cjk_any=mean("cjk_any"), reply_target=mean("reply_target"))
    elif name in ("mgsm_zh", "practice_zh"):
        s.update(acc=mean("correct"), boxed=mean("boxed"), zh_reply=mean("zh_reply"), answer_en=mean("answer_en"),
                 think_zh=mean("think_zh"), han_frac=mean("han_frac"), reply_target=mean("reply_target"),
                 think_lang_share=mean_def("think_lang_share"), mix=mean("mix"))
    else:
        s.update(acc=mean("correct"), boxed=mean("boxed"), mix=mean("mix"), han_run=mean("han_run"),
                 han_answer=mean("han_answer"), lid_zh=mean("lid_zh"), reply_target=mean("reply_target"),
                 think_lang_share=mean_def("think_lang_share"), think_script_share=mean_def("think_script_share"),
                 han_frac=mean("han_frac"))
        if name == "math500":
            s["acc_strict"] = mean("correct_strict")
    return s


# ---------------------------------------------------------------- chat KL

def chat_refs(arm, items):
    """Base model's sampled answers to the chat prompts (KL_TOKENS tokens), generated once per model and cached."""
    path = C.DATA / f"chat_ref_{C.model_tag(arm.a.model)}.json"
    refs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [it for it in items if it["id"] not in refs]
    if todo:
        undo = C.apply_edit(arm.model, arm.edit_orig) if arm.edit_orig else None
        try:
            for s in range(0, len(todo), 75):
                chunk = todo[s:s + 75]
                prompts = [C.chat_prompt(arm.tok, C.user_text(it)) for it in chunk]
                enc = arm.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                with C.gpu(f"langmix/eval.py chat refs {s}"):
                    torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC)
                    arm.model.cuda()
                    torch.manual_seed(C.SEED0 + s)
                    with torch.no_grad():
                        g = arm.model.generate(**{k: v.cuda() for k, v in enc.items()}, max_new_tokens=KL_TOKENS,
                                               pad_token_id=arm.tok.pad_token_id, **C.SAMPLING)
                    g = g[:, enc.input_ids.shape[1]:].cpu()
                    arm.model.cpu()
                for it, ids in zip(chunk, g):
                    ids = ids.tolist()
                    n = next((i for i, x in enumerate(ids) if x in (arm.tok.pad_token_id, arm.tok.eos_token_id)), len(ids))
                    refs[it["id"]] = ids[:max(n, 1)]
                path.write_text(json.dumps(refs), encoding="utf-8")
        finally:
            if undo is not None:
                C.apply_edit(arm.model, undo)
    return refs


@torch.no_grad()
def run_chat(arm, odir):
    a = arm.a
    items = D.load("chat")[: a.limit]
    refs = chat_refs(arm, items)
    cjk = torch.zeros(arm.model.config.vocab_size, dtype=torch.bool)
    cjk[torch.tensor(C.cjk_token_ids(arm.tok))] = True
    def logprobs(it, side):
        comp = refs[it["id"]]
        prompt = C.chat_prompt(arm.tok, C.user_text(it), arm.instr if side == "arm" else None, a.instr_role)
        p = arm.tok(prompt, add_special_tokens=False).input_ids
        ids = torch.tensor([p + comp]).cuda()
        if side == "arm" and arm.steer is not None:
            st = arm.steer
            with C.Steer(arm.model, st["layer"], st["vec"], arm.alphas([it])):
                logits = arm.model(input_ids=ids).logits[0]
        else:
            logits = arm.model(input_ids=ids).logits[0]
        logits = logits[len(p) - 1: len(p) - 1 + len(comp)].float()
        if side == "arm" and arm.ban is not None:
            logits = logits.masked_fill(cjk[: logits.shape[-1]], float("-inf"))
        return torch.log_softmax(logits, -1)

    rows, G = [], 4  # texts per group; base log-probs of a group stay on the GPU in fp32 (~0.6 GB)
    with C.gpu(f"langmix/eval.py {a.arm} chat KL"):
        torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC)
        t = time.time()
        arm.model.cuda()
        cjk = cjk.cuda()
        for g in range(0, len(items), G):
            grp = items[g:g + G]
            if arm.edit_orig:
                C.apply_edit(arm.model, arm.edit_orig)
            base = [logprobs(it, "base") for it in grp]
            if arm.edit_orig:
                C.apply_edit(arm.model, arm.edit)
            for it, blp in zip(grp, base):
                lp = logprobs(it, "arm")
                kl = (blp.exp() * (blp - lp.clamp_min(KL_FLOOR))).sum(-1)
                rows.append(dict(id=it["id"], n=len(refs[it["id"]]), kl=kl.mean().item(), kl_max=kl.max().item(),
                                 top1_agree=(blp.argmax(-1) == lp.argmax(-1)).float().mean().item(),
                                 p_cjk_base=blp.exp()[:, cjk[: blp.shape[-1]]].sum(-1).mean().item(),
                                 p_cjk_arm=lp.exp()[:, cjk[: lp.shape[-1]]].sum(-1).mean().item()))
            del base
        arm.model.cpu()
        secs = time.time() - t
    with open(odir / "chat.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    return dict(n=n, kl=round(sum(r["kl"] for r in rows) / n, 5), kl_max=round(max(r["kl_max"] for r in rows), 3),
                top1_agree=round(sum(r["top1_agree"] for r in rows) / n, 4),
                p_cjk_base=round(sum(r["p_cjk_base"] for r in rows) / n, 6),
                p_cjk_arm=round(sum(r["p_cjk_arm"] for r in rows) / n, 6), gpu_min=round(secs / 60, 1))


# ---------------------------------------------------------------- main

def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--instr", choices=sorted(C.INSTRUCTIONS), default=None)
    ap.add_argument("--instr-text", default=None)
    ap.add_argument("--instr-role", choices=["user", "system"], default="user")
    ap.add_argument("--ban-cjk", action="store_true")
    ap.add_argument("--steer", default=None)
    ap.add_argument("--steer-alpha", type=float, default=1.0)
    ap.add_argument("--steer-mode", choices=["all", "cond", "cond0"], default="cond")
    ap.add_argument("--target-lang", default=None,
                    help="language every answer should be in (default: each prompt's own language)")
    ap.add_argument("--suites", default=",".join(ALL))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--max-new", type=int, default=C.MAX_NEW)
    ap.add_argument("--no-reuse-check", action="store_true")
    ap.add_argument("--out-root", default=str(EVAL))
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    global TARGET
    TARGET = a.target_lang
    odir = Path(a.out_root) / a.arm
    odir.mkdir(parents=True, exist_ok=True)
    tok = C.load_tok(a.model)
    model = C.load_model(a.model)
    arm = Arm(a, model, tok)
    cfg = arm.config()
    cpath = odir / "arm.json"
    if cpath.exists():
        old = json.loads(cpath.read_text())
        diff = {k for k in cfg if k != "batch" and old.get(k) != cfg[k]}
        if diff:
            sys.exit(f"{odir} was produced by a different arm config ({sorted(diff)}); use another --arm name")
    cpath.write_text(json.dumps(cfg, indent=1))
    spath = odir / "summary.json"
    summary = json.loads(spath.read_text()) if spath.exists() else {}
    summary["config"] = cfg
    for name in a.suites.split(","):
        t = time.time()
        print(f"[{a.arm}] {name}", flush=True)
        if name == "chat":
            s = run_chat(arm, odir)
        else:
            s = summarize_gen(name, run_gen_suite(arm, name, odir))
        s["limit"] = a.limit
        s["wall_min"] = round((time.time() - t) / 60, 1)
        summary[name] = s
        spath.write_text(json.dumps(summary, indent=1))
        print(f"  {json.dumps(s)}", flush=True)


if __name__ == "__main__":
    main()
