"""Extra arm for langmix: the blind agents' recipe (old run, langmix/recipe_script.py), ported to patchkit's output contract.

    python -m tasks._langmix_recipe [--task langmix] [--alpha 0.25] [--k 256]
    python -m patchkit.run eval langmix --arms gate --tag recipe        # -> results/langmix/gate_recipe.json

A continuous, context-gated logit penalty on the lm_head rows of every CJK token (not a thresholded neuron).
h = the lm_head input (after the final norm) at reply positions of the practice base replies, teacher-forced:
  native = mean h over de/ru/th reply positions whose next token is not CJK (leaking replies: before the first leak)
  zh     = mean h over zh reply positions whose next token is CJK;  en = mean h over en reply positions
  u      = unit(native - zh) with its component along zh removed
  u_k    = u on its top-k dims by |u_j (native_j - zh_j)|, projected off span{zh, en} on those dims
  edit   = lm_head rows of every CJK token -= alpha * u_k   -> each CJK logit moves by -alpha * u_k.h
alpha = 0.25 and k = 256 are the old recipe's fixed values (session 1's choices); nothing is tuned here. Differences
from the old script: practice = this task's practice items and greedy replies (the old one used the agents' mined
practice set with sampled replies), and leaking replies contribute only their positions before the first leak.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patchkit import llm as L  # noqa: E402
from patchkit import run as R  # noqa: E402
from patchkit import task as T  # noqa: E402
from tasks import _langmix_text as X  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="langmix")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    task = T.get(a.task)
    c = R.Ctx(task)
    tok, model = c.tok, c.model
    items, base = c.items("practice"), c.base()
    mask = X.cjk_mask(tok, model.config.vocab_size)
    d = model.config.hidden_size
    acc = {k: [torch.zeros(d, dtype=torch.float64), 0] for k in ("native", "zh", "en")}
    for x in items:
        p = task.render(tok, x)
        n_p = len(tok(p, add_special_tokens=False).input_ids)
        ids = tok(p + base[x["id"]]["base"], add_special_tokens=False).input_ids
        h = model.model(input_ids=torch.tensor([ids], device=L.DEV)).last_hidden_state[0].float().cpu()
        pos = torch.arange(n_p - 1, len(ids) - 1)
        nxt = mask[torch.tensor(ids)[pos + 1]]
        if x["lang"] in ("de", "ru", "th"):
            on = nxt.nonzero()
            sel = pos[: int(on[0])] if len(on) else pos
            key = "native"
        elif x["lang"] == "zh":
            sel, key = pos[nxt], "zh"
        else:
            sel, key = pos, "en"
        acc[key][0] += h[sel].double().sum(0)
        acc[key][1] += len(sel)
    native, zh, en = [(acc[k][0] / acc[k][1]).float() for k in ("native", "zh", "en")]
    dv = native - zh
    dv = dv - (dv @ zh) / (zh @ zh) * zh
    u = dv / dv.norm()
    idx = (u * (native - zh)).abs().topk(a.k).indices
    B = torch.stack([zh[idx], en[idx]], 1)
    uk = torch.zeros_like(u)
    uk[idx] = u[idx] - B @ torch.linalg.lstsq(B, u[idx, None]).solution[:, 0]
    name = L.pname(model, model.get_output_embeddings().weight)
    W0 = model.get_output_embeddings().weight.detach().cpu()
    W = W0.float().clone()
    W[mask[: W.shape[0]]] -= a.alpha * uk
    W = W.to(W0.dtype)
    meta = dict(method="recipe (continuous context-gated CJK logit penalty, lm_head rows)", alpha=a.alpha, k=a.k,
                entries_changed=int((W != W0).sum()), n_cjk_rows=int(mask.sum()),
                positions=dict(native=acc["native"][1], zh=acc["zh"][1], en=acc["en"][1]),
                uk_dot=dict(native=round(float(native @ uk), 3), zh=round(float(zh @ uk), 3), en=round(float(en @ uk), 3)),
                n_items=len(items), fit_seconds=round(time.time() - t0, 1))
    L.save_edit({name: W.contiguous()}, c.runs / "edits" / "gate_recipe.safetensors", meta)
    print("recipe:", json.dumps(meta))


if __name__ == "__main__":
    main()
