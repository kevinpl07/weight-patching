"""Anatomy of a fitted gate, for the blog figures.

    python experiments/gate_anatomy.py entity [--tag ""]

Writes results/<task>/anatomy.json:
  - scores: the gate neuron's pre-activation in threshold units, (gate_row . x) / g = d.x - theta, on HELD-OUT
    decision sites (failure cases vs good cases) and on ordinary reply positions (teacher-forced base replies).
    It fires where the score is > 0. This is the "the model already knows" histogram.
  - write: the tokens whose logits the neuron's write vector raises / lowers most (per unit of activation).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from patchkit import llm as L  # noqa: E402
from patchkit import methods as M  # noqa: E402
from patchkit import task as T  # noqa: E402
from patchkit.run import Ctx  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--model")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-content-items", type=int, default=60)
    a = ap.parse_args()
    task = T.get(a.task)
    if a.model:
        task.model, task.name = a.model, f"{task.name}__{a.model.split('/')[-1]}"
    c = Ctx(task)
    tok, model = c.tok, c.model
    sfx = f"_{a.tag}" if a.tag else ""
    meta = json.loads((c.runs / "edits" / f"gate{sfx}.json").read_text())
    edit = L.load_edit(c.runs / "edits" / f"gate{sfx}.safetensors")
    Lr, unit = meta["layer"], meta["unit"]
    mlp = L.text_model(model).layers[Lr].mlp
    gname = L.pname(model, mlp.gate_proj.weight)
    dname = L.pname(model, mlp.down_proj.weight)
    grow = edit[gname][unit].float()
    dcol = edit[dname][:, unit].float()
    held = c.items("heldout")
    base_out = {r["id"]: r["text"] for r in L.read_jsonl(c.res / "outputs" / "base.jsonl")}
    sites = []
    for x in held:
        lab = task.positive(x)
        if lab is None or x["id"] not in base_out:
            continue
        sites += [dict(s, id=x["id"]) for s in task.probe_sites(tok, x, base_out[x["id"]])]
    rng = random.Random(0)
    con = [s for s in sites if s["kind"] == "content"]
    con = rng.sample(con, min(a.n_content_items, len(con)))
    dec = [s for s in sites if s["kind"] == "decision"]
    X, lab, kind = M.capture_sites(model, tok, dec + con, [Lr])
    sc = (X[Lr] @ grow.cpu()) / task.gate_g
    out = dict(task=task.name, model=task.model, layer=Lr, unit=unit, theta=meta["theta"],
               fail=[round(v, 3) for v, l_, k in zip(sc.tolist(), lab.tolist(), kind) if k == "decision" and l_ == 1],
               good=[round(v, 3) for v, l_, k in zip(sc.tolist(), lab.tolist(), kind) if k == "decision" and l_ == 0],
               content=[round(v, 3) for v, k in zip(sc.tolist(), kind) if k == "content"])
    # write vector through the final norm + unembedding (direction only, per unit activation)
    tm = L.text_model(model)
    v = dcol.to(L.DEV)
    if L.is_gemma(model):
        v = v * (1 + tm.layers[Lr].post_feedforward_layernorm.weight.float())
    E = model.get_output_embeddings().weight.float()
    eff = E @ (v * ((1 + tm.norm.weight.float()) if L.is_gemma(model) else tm.norm.weight.float()))
    top = eff.topk(15)
    bot = (-eff).topk(15)
    out["write_up"] = [(tok.decode([i]), round(s, 2)) for s, i in zip(top.values.tolist(), top.indices.tolist())]
    out["write_down"] = [(tok.decode([i]), round(-s, 2)) for s, i in zip(bot.values.tolist(), bot.indices.tolist())]
    out["fire_rate"] = dict(fail=sum(s > 0 for s in out["fail"]) / max(1, len(out["fail"])),
                            good=sum(s > 0 for s in out["good"]) / max(1, len(out["good"])),
                            content=sum(s > 0 for s in out["content"]) / max(1, len(out["content"])))
    (c.res / "anatomy.json").write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps(out["fire_rate"]), out["write_up"][:8])


if __name__ == "__main__":
    main()
