"""Quick-eval table across the 100-step arms (runs/grpo/arm*_*_100/evals.jsonl).

    python rl/evals_table.py            # every run, every eval step
    python rl/evals_table.py --last     # only the newest eval line of each run
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLS = [("symT1_acc", "sampled"), ("symT1_trunc", "trunc"), ("symT1_len", "len"), ("sym_acc", "greedy"),
        ("ifeval_strict", "IFEval"), ("ifeval_loop", "IF loop")]


def load():
    runs = {}
    for f in sorted((ROOT / "runs" / "grpo").glob("arm*_*_100/evals.jsonl")):
        runs[f.parent.name] = {r["step"]: r for r in map(json.loads, f.read_text().splitlines())}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", action="store_true")
    a = ap.parse_args()
    runs = load()
    print(f"{'run':14s}{'step':>5s} " + " ".join(f"{h:>8s}" for _, h in COLS) + "   sampled vs A same step")
    for name, ev in runs.items():
        ref = runs.get(f"armA_s{name.split('_s')[1]}", runs.get("armA_s0_100", {}))
        for step in ([max(ev)] if a.last else sorted(ev)):
            r = ev[step]
            cells = [f"{r[k]:8.0f}" if k.endswith("len") else f"{100 * r[k]:7.1f}%" for k, _ in COLS]
            vs = f"{100 * (r['symT1_acc'] - ref[step]['symT1_acc']):+.1f}" if step in ref and not name.startswith("armA") else ""
            print(f"{name:14s}{step:5d} " + " ".join(cells) + f"   {vs}")


if __name__ == "__main__":
    main()
