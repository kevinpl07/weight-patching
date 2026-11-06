"""Matched-point table for amendments A1/A2 (training checkpoints selected without eval data) next to the agent edits.
  python vision_spurious/summarize_curves.py              (A1: curves/, rule "cv")
  python vision_spurious/summarize_curves.py curves_in    (A2: curves_in/, rule "prac")"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

E = C.RUNS / "eval"
CU = C.RUNS / (sys.argv[1] if len(sys.argv) > 1 else "curves")
VARS = sorted(p.stem for p in CU.glob("*.json"))
AGENT_DEV = dict(s1=0.234, s2=0.165, s3=0.173)


def main():
    base = json.loads((E / "none.json").read_text())["summary"]
    c1 = lambda h: h["dev_spur_hit"] <= 0.5 * base["dev_spur_hit"] and h["dev_spur_ap"] >= base["dev_spur_ap"] + 0.10  # noqa: E731
    c2 = lambda h: h["top1"] >= base["top1"] - 0.005 and h["top1_dev_classes"] >= base["top1_dev_classes"] - 0.03  # noqa: E731
    row = lambda name, step, ts, dh, h: (  # noqa: E731
        f"| {name} | {step} | {ts} | {dh} | {100 * h['dev_spur_hit']:.1f}% | {h['dev_spur_ap']:.3f} | "
        f"{100 * (h['top1'] - base['top1']):+.2f} | {100 * h['top1_dev_classes']:.1f}% | "
        f"{100 * h['unseen_spur_hit']:.1f}% | {'yes' if c1(h) else 'no'} / {'yes' if c2(h) else 'no'} |")
    print("| arm | step | train s | dev hit (in-sample) | held-out hit | held-out AP | Δ top-1 | top-1 dev cls | unseen hit | cond 1 / 2 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(row("none", 0, "-", "0.530", base))
    for s in ("s1", "s2", "s3"):
        print(row(f"**agent {s}**", "-", "-", f"{AGENT_DEV[s]:.3f}", json.loads((E / f"agent_{s}.json").read_text())["summary"]))
        for v in VARS:
            J = json.loads((CU / f"{v}.json").read_text())["selected"]
            sel = J.get(s) or J.get(f"match_{s}")
            if sel:
                dh = sel["dev"]["hit"] if "dev" in sel else sel["dev_hit"]
                print(row(f"{v} @ match {s}", sel["step"], sel["train_s"], f"{dh:.3f}", sel["heldout"]))
            else:
                print(f"| {v} @ match {s} | not reached | | | | | | | | |")
    for v in VARS:
        J = json.loads((CU / f"{v}.json").read_text())["selected"]
        lab = "cv" if "cv" in J else "prac"
        sel = J[lab]
        dh = sel["dev"]["hit"] if "dev" in sel else sel["dev_hit"]
        print(row(f"{v} @ {'3-fold CV' if lab == 'cv' else 'practitioner rule'}", sel["step"], sel["train_s"], f"{dh:.3f}", sel["heldout"]))


if __name__ == "__main__":
    main()
