"""Where does an edit change the model's uncertainty? Teacher-force the same answers through the base and edited
models and compare next-token entropy at "hedge points" (the next token starts a detour: Wait, Alternatively,
maybe, ...) against all other answer positions. A temperature change would lower entropy everywhere; a targeted
"commit" mechanism would lower it mainly at hedge points.

    python rl/mech_check.py --edited runs/agent_dry/model_edit --answers runs/eval/base/symbolic_sample.jsonl.gz --n 200
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import chat, symbolic  # noqa: E402
from grpo import MODEL, SYSTEM  # noqa: E402

HEDGES = {"wait", "alternatively", "however", "maybe", "perhaps", "actually", "hmm", "possibly", "might",
          "likely", "hold", "interpret", "interpretation", "re", "reconsider", "assuming", "assume", "unless"}


def stats(logits, targets, T=1.0):
    lp = torch.log_softmax(logits.float() / T, -1)
    ent = -(lp.exp() * lp).sum(-1)
    return ent, lp.gather(-1, targets[:, None])[:, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edited", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.set_per_process_memory_fraction(0.8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    edit = AutoModelForCausalLM.from_pretrained(a.edited, dtype=torch.bfloat16).cuda().eval()
    q = {p["id"]: p["question"] for p in symbolic()}
    recs = [json.loads(l) for l in gzip.open(a.answers, "rt", encoding="utf-8")][:a.n]
    is_hedge = {}
    acc = {k: [] for k in ["hb", "he", "ht8", "lb", "le", "lt8", "hedge"]}
    with torch.no_grad():
        for r in recs:
            prompt = chat(tok, q[r["id"]], SYSTEM)
            P = len(tok(prompt).input_ids)
            ids = tok(prompt + r["completion"], return_tensors="pt").input_ids.cuda()
            tgt = ids[0, P:]
            lb = base(input_ids=ids).logits[0, P - 1:-1]
            le = edit(input_ids=ids).logits[0, P - 1:-1]
            hb, lpb = stats(lb, tgt); he, lpe = stats(le, tgt); ht8, lpt8 = stats(lb, tgt, 0.8)
            for t in tgt.tolist():
                if t not in is_hedge:
                    w = tok.decode([t]).strip().lower().lstrip("*(\"'")
                    is_hedge[t] = w in HEDGES
            acc["hedge"].append(torch.tensor([is_hedge[t] for t in tgt.tolist()]))
            for k, v in zip(["hb", "he", "ht8", "lb", "le", "lt8"], [hb, he, ht8, lpb, lpe, lpt8]):
                acc[k].append(v.cpu())
    c = {k: torch.cat(v) for k, v in acc.items()}
    m = c["hedge"]
    # positions *before* a hedge token: the decision point is where the hedge token is predicted, i.e. target = hedge
    for name, sel in [("hedge decisions", m), ("all other tokens", ~m)]:
        n = int(sel.sum())
        print(f"{name:18s} n={n:6d}  entropy base {c['hb'][sel].mean():.3f}  edit {c['he'][sel].mean():.3f} "
              f"(d {(c['he'] - c['hb'])[sel].mean():+.3f})  base@T0.8 {c['ht8'][sel].mean():.3f} "
              f"(d {(c['ht8'] - c['hb'])[sel].mean():+.3f})  |  logp(actual) d edit {(c['le'] - c['lb'])[sel].mean():+.3f}  "
              f"d T0.8 {(c['lt8'] - c['lb'])[sel].mean():+.3f}")
    hi = c["hb"] > 1.0
    for name, sel in [("uncertain (H>1), non-hedge", hi & ~m), ("confident (H<=1)", ~hi & ~m)]:
        print(f"{name:27s} n={int(sel.sum()):6d}  d entropy edit {(c['he'] - c['hb'])[sel].mean():+.3f}  "
              f"T0.8 {(c['ht8'] - c['hb'])[sel].mean():+.3f}")


if __name__ == "__main__":
    main()
