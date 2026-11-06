"""Diagnostic for langmix: what the gate neuron does at ordinary positions (off-state leakage through SiLU's negative
tail). Practice data only.

    python -m tasks._langmix_gate_diag [--tag ""]

Applies runs/langmix/edits/gate{_tag}.safetensors, teacher-forces practice base replies (zh, en, non-leaking de/ru/th
and leaking ones), and records the gate unit's activation a = silu(gate_proj . x) * (up_proj . x) at every reply
position. The neuron writes a * down_proj[:, unit] (norm = gain) into the residual, so the write is a * gain along v.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patchkit import llm as L  # noqa: E402
from patchkit import run as R  # noqa: E402
from patchkit import task as T  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-group", type=int, default=24)
    a = ap.parse_args()
    task = T.get("langmix")
    c = R.Ctx(task)
    tok, model = c.tok, c.model
    p = c.runs / "edits" / f"gate{'_' + a.tag if a.tag else ''}.safetensors"
    meta = json.loads(p.with_suffix(".json").read_text())
    unit, Lr = meta["unit"], meta["layer"]
    orig = L.apply_edit(model, L.load_edit(p))
    mlp = L.text_model(model).layers[Lr].mlp
    gain = float(mlp.down_proj.weight[:, unit].float().norm())
    cur = {}
    h = mlp.down_proj.register_forward_hook(lambda m, i, o: cur.__setitem__("a", i[0][0, :, unit].float().cpu()))
    base = c.base()
    groups = {"zh": [], "en": [], "de/ru/th clean": [], "de/ru/th leaking": []}
    for x in c.items("practice"):
        k = x["lang"] if x["lang"] in ("zh", "en") else ("de/ru/th leaking" if x.get("leaks") else "de/ru/th clean")
        if len(groups[k]) < a.per_group:
            groups[k].append(x)
    out = {}
    try:
        for k, xs in groups.items():
            acts = []
            for x in xs:
                pr = task.render(tok, x)
                n_p = len(tok(pr, add_special_tokens=False).input_ids)
                ids = tok(pr + base[x["id"]]["base"], add_special_tokens=False).input_ids
                model(input_ids=torch.tensor([ids], device=L.DEV))
                acts.append(cur["a"][n_p - 1:])
            A = torch.cat(acts)
            out[k] = dict(positions=len(A), mean_act=round(A.mean().item(), 4),
                          mean_write_along_v=round(gain * A.mean().item(), 2),
                          frac_positive=round((A > 0).float().mean().item(), 4),
                          q01=round(A.quantile(0.01).item(), 4), q50=round(A.median().item(), 4),
                          max=round(A.max().item(), 3))
    finally:
        h.remove()
        L.apply_edit(model, orig)
    res = dict(edit=p.name, unit=unit, layer=Lr, down_norm=round(gain, 1), theta=meta["theta"], groups=out)
    (c.res / f"gate_diag{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
