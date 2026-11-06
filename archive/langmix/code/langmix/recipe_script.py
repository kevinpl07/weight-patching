"""Arm 7, "it's just a script": the agents' CJK gate on lm_head (sessions 1 and 2), built in closed form from practice
data only.

  python langmix/recipe_script.py [--alpha 0.25] [--k 256] [--no-check]

Data: runs/langmix/practice/practice.json (mix / clean: de, ru, th; zh; en), the stored base answers teacher-forced
under the scoring prompt. h = the lm_head input (after the final norm) at every answer position; "CJK next" = the next
token is in common.cjk_token_ids (31,778 tokens). Texts are cropped as in session 1: mix answers up to 200 characters
after their third Han character, all others to their first 1,500 characters.
Fixed rules (session 1's choices; nothing tuned on held-out data, no dose sweep):

  native = mean h over mix and clean answers where the next token is not CJK (de/ru/th context)
  zh     = mean h over zh answers where the next token is CJK;  en = mean h over en answers
  u      = unit(native - zh), with its component along zh removed
  u_k    = u on its top-k dims by |u_j (native_j - zh_j)|, then projected off span{zh, en} on those dims (k = 256)
  edit   = lm_head rows of every CJK token -= alpha * u_k (alpha = 0.25), so each CJK logit moves by -alpha * u_k.h:
           large in de/ru/th context, about 0 on average in Chinese and English. 31,778 x 256 = 8.14M entries
           (budget 8,885,440).

Outputs: runs/langmix/recipe/recipe_gate.safetensors and runs/langmix/recipe/arm_recipe_gate.json (the eval.py command).
The verification block after the recipe (not part of it) compares the edit with the agents' submitted edits, scores
CJK probability per context offline, and resamples 64 tokens from practice answers (base vs edited, paired seeds).
"""
import argparse
import json
import time
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (also puts the repo root on sys.path)
import metrics as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.25)
ap.add_argument("--k", type=int, default=256)
ap.add_argument("--out", default=str(C.RUNS / "recipe" / "recipe_gate.safetensors"))
ap.add_argument("--no-check", action="store_true")
a = ap.parse_args()

# ---------------------------------------------------------------- logic (30 lines, prints included)
P, tok, model = json.loads((C.RUNS / "practice" / "practice.json").read_text(encoding="utf-8")), C.load_tok(), C.load_model()
cjk = torch.tensor(C.cjk_token_ids(tok)); is_cjk = torch.zeros(model.config.vocab_size, dtype=torch.bool); is_cjk[cjk] = True
def crop(s, t):  # session 1's crop
    han = [m.start() for m in M.HAN.finditer(t)]
    return t[:han[min(2, len(han) - 1)] + 200] if s == "mix" and han else t[:1500]
rows = []  # (set, item, token ids, prompt length, h [answer positions, d] fp16, next-is-CJK [answer positions])
with C.gpu("langmix recipe_script capture"), torch.no_grad():  # one hold, ~300 short teacher-forced texts
    torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC); model.cuda()
    for s, items in P.items():
        for it in items:
            pre = C.chat_prompt(tok, C.user_text(it)); pl = len(tok(pre, add_special_tokens=False).input_ids)
            ids = tok(pre + crop(s, it["completion"]), add_special_tokens=False, return_tensors="pt").input_ids[0]
            h = model.model(input_ids=ids[None].cuda()).last_hidden_state[0, pl - 1:-1]  # = lm_head input (after norm)
            rows.append((s, it, ids, pl, h.half().cpu(), is_cjk[ids[pl:]]))
    model.cpu(); torch.cuda.empty_cache()
def mean(sets, sel):  # mean h over the selected positions of the given practice sets
    return sum(h[sel(n)].float().sum(0) for s, _, _, _, h, n in rows if s in sets) / sum(int(sel(n).sum()) for s, *_, n in rows if s in sets)
native, zh, en = mean(("mix", "clean"), lambda n: ~n), mean(("zh",), lambda n: n), mean(("en",), torch.ones_like)
d = native - zh; d = d - (d @ zh) / (zh @ zh) * zh; u = d / d.norm()
idx = (u * (native - zh)).abs().topk(a.k).indices
B = torch.stack([zh[idx], en[idx]], 1); uk = torch.zeros_like(u)
uk[idx] = u[idx] - B @ torch.linalg.lstsq(B, u[idx, None]).solution[:, 0]  # plain projection off span{zh, en}
W0 = model.lm_head.weight.detach(); Wl = W0.float().clone(); Wl[cjk] -= a.alpha * uk
Path(a.out).parent.mkdir(parents=True, exist_ok=True); save_file({"lm_head.weight": Wl.to(torch.bfloat16).contiguous()}, a.out)
arm = dict(arm="recipe_gate", edit=Path(a.out).relative_to(C.ROOT).as_posix(), suites=["mgsm_de", "mgsm_ru", "mgsm_th", "mgsm_en", "mgsm_zh", "ifeval", "chat"])
arm["command"] = f".venv/Scripts/python langmix/eval.py --arm {arm['arm']} --edit {arm['edit']} --suites {','.join(arm['suites'])}"
(Path(a.out).parent / "arm_recipe_gate.json").write_text(json.dumps(arm, indent=1)); print(arm["command"])
print(f"positions: native {sum(int((~n).sum()) for s, *_, n in rows if s in ('mix', 'clean'))}, zh CJK-next {sum(int(n.sum()) for s, *_, n in rows if s == 'zh')}, "
      f"en {sum(len(n) for s, *_, n in rows if s == 'en')}; |u_k| {uk.norm():.3f}; entries changed {int((Wl.to(torch.bfloat16) != W0).sum()):,}; "
      f"delta norm {float((Wl.to(torch.bfloat16).float() - W0.float()).norm()):.1f}; u_k.h means: native {native @ uk:.2f}, zh {zh @ uk:.2f}, en {en @ uk:.2f}")

# ---------------------------------------------------------------- verification (not part of the recipe)
if not a.no_check:
    from safetensors import safe_open
    Wr = Wl.to(torch.bfloat16); del Wl
    print("\nagent edit vs recipe (lm_head deltas): entries | rows (overlap with the recipe's CJK rows) | columns "
          "(overlap with the recipe's top-k) | Frobenius cos | cos of the mean row delta | norm")
    for s in (1, 2):
        with safe_open(str(C.RUNS / "sessions" / f"session_{s}" / "submitted_edit.safetensors"), "pt") as fh:
            Wa = fh.get_tensor("lm_head.weight")
        dot = na = nr = 0.0; ent = 0; rows_a, cols_a, ra, rr = [], torch.zeros(W0.shape[1], dtype=torch.bool), 0, 0
        for b in range(0, W0.shape[0], 16384):
            Da, Dr = Wa[b:b + 16384].float() - W0[b:b + 16384].float(), Wr[b:b + 16384].float() - W0[b:b + 16384].float()
            dot += float((Da * Dr).sum()); na += float(Da.pow(2).sum()); nr += float(Dr.pow(2).sum()); ent += int((Da != 0).sum())
            rows_a += (torch.nonzero(Da.abs().sum(1)).flatten() + b).tolist(); cols_a |= Da.abs().sum(0) > 0
            ra, rr = ra + Da.sum(0), rr + Dr.sum(0)
        ov = len(set(rows_a) & set(cjk.tolist()))
        print(f"s{s}: {ent:,} | {len(rows_a):,} ({ov:,} of {len(cjk):,}) | {int(cols_a.sum())} ({int(cols_a[idx].sum())} of {a.k}) | "
              f"{dot / (na * nr) ** 0.5:+.3f} | {float(ra @ rr / (ra.norm() * rr.norm())):+.3f} | {na ** 0.5:.1f} (recipe {nr ** 0.5:.1f})")
        del Wa
    # offline: mean CJK probability per context, base vs edited lm_head (up to 600 positions per group, fixed seed)
    g = torch.Generator().manual_seed(0)
    groups = {"leak onset (1st CJK, mix)": torch.stack([h[int(n.nonzero()[0])] for s, _, _, _, h, n in rows if s == "mix" and n.any()]),
              "leak, all CJK-next (mix)": torch.cat([h[n] for s, _, _, _, h, n in rows if s == "mix"]),
              "mix, not CJK-next": torch.cat([h[~n] for s, _, _, _, h, n in rows if s == "mix"]),
              "clean": torch.cat([h for s, _, _, _, h, n in rows if s == "clean"]),
              "zh, CJK-next": torch.cat([h[n] for s, _, _, _, h, n in rows if s == "zh"]),
              "zh, other": torch.cat([h[~n] for s, _, _, _, h, n in rows if s == "zh"]),
              "en": torch.cat([h for s, _, _, _, h, n in rows if s == "en"])}
    groups = {k: X[torch.randperm(len(X), generator=g)[:600]] for k, X in groups.items()}
    # resampling: continue practice answers for 64 tokens at the scoring sampling settings, base vs edited, same seeds.
    # mix: from 32 tokens before the stored answer's first CJK token (first 48 answers where that is >= 32 tokens in);
    # clean / zh: from answer token 256 (first 24 answers longer than 320 tokens).
    pick = lambda s, f, k: [ids[:pl + f(n)] for t, _, ids, pl, _, n in rows if t == s and f(n) is not None][:k]  # noqa: E731
    pre = {"mix": pick("mix", lambda n: int(n.nonzero()[0]) - 32 if n.any() and int(n.nonzero()[0]) >= 32 else None, 48),
           "clean": pick("clean", lambda n: 256 if len(n) > 320 else None, 24), "zh": pick("zh", lambda n: 256 if len(n) > 320 else None, 24)}
    out, lines = {}, []
    with C.gpu("langmix recipe_script check"), torch.no_grad():  # one hold for the whole check (~1-2 min)
        torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC); t0 = time.time()
        Wb, We, m = W0.cuda(), Wr.cuda(), is_cjk.cuda()
        for name, X in groups.items():
            X = X.cuda().to(torch.bfloat16)
            pb, pe = [torch.softmax((X @ w.T).float(), -1)[:, m].sum(-1).mean().item() for w in (Wb, We)]
            lines.append(f"  {name:28s} n={len(X):4d}  {pb:.4f} -> {pe:.4f}  [{-a.alpha * float((X.float() @ uk.cuda()).mean()):+.2f}]")
        del Wb, We; torch.cuda.empty_cache(); model.cuda()
        for side in ("base", "edited"):
            if side == "edited":
                model.lm_head.weight.data.copy_(Wr.cuda())
            for s, ps in pre.items():
                res = []
                for b in range(0, len(ps), 24):
                    batch = ps[b:b + 24]; Lmax = max(len(x) for x in batch)
                    ids = torch.stack([torch.cat([torch.full((Lmax - len(x),), tok.pad_token_id), x]) for x in batch]).cuda()
                    att = torch.stack([torch.cat([torch.zeros(Lmax - len(x)), torch.ones(len(x))]) for x in batch]).long().cuda()
                    torch.manual_seed(1000 + b)
                    gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=64, pad_token_id=tok.pad_token_id, **C.SAMPLING)
                    res += [C.decode_row(tok, r, 64)[0] for r in gen[:, Lmax:].cpu()]
                out[(side, s)] = (f"{sum(bool(M.HAN.search(t)) for t in res)}/{len(res)}" if s != "zh" else f"{sum(M.han_frac(t) for t in res) / len(res):.3f} (n={len(res)})")
        model.lm_head.weight.data.copy_(W0.cuda()); model.cpu(); torch.cuda.empty_cache(); secs = time.time() - t0
    print("\noffline mean P(next token is CJK), base -> edited  [mean CJK logit shift -alpha u_k.h]\n" + "\n".join(lines))
    print("\nresampling 64 tokens (paired seeds): mix/clean = continuations with a Han character; zh = mean Han share of letters")
    for side in ("base", "edited"):
        print(f"  {side:7s} " + "  ".join(f"{s} {out[(side, s)]}" for s in pre))
    print(f"check GPU seconds inside the lock: {secs:.0f}")
