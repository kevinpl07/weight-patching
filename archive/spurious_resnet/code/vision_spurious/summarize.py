"""Table of all evaluated arms + pass-bar check (plan: PLAN_2026-11-04_spurious_repair.md).
  python vision_spurious/summarize.py"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

E = C.RUNS / "eval"
ARMS = ["none", "agent_s1", "agent_s2", "agent_s3", "random_same_s1", "random_same_s2", "random_same_s3",
        "random_count_s1", "random_count_s2", "random_count_s3", "proj", "spufix_like", "llr", "ft_l4"]


def main():
    d = torch.load(C.DERIVED / "eval.pt")
    clean = torch.tensor([k == "clean" for k in d["kind"]])
    y = d["labels"][clean]
    S = {a: json.loads((E / f"{a}.json").read_text())["summary"] for a in ARMS if (E / f"{a}.json").exists()}
    base = S["none"]
    c0 = (torch.load(E / "none_logits.pt")[clean].float().argmax(1) == y).float()
    print("| arm | entries | dev hit | dev AP | dev AUC | top-1 | Δ top-1 (paired SE) | top-1 dev cls | unseen hit | unseen AP | top-1 unseen cls |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a, s in S.items():
        c = (torch.load(E / f"{a}_logits.pt")[clean].float().argmax(1) == y).float()
        diff = c - c0
        se = diff.std().item() / len(diff) ** 0.5
        print(f"| {a} | {s['entries_changed']:,} | {s['dev_spur_hit']:.3f} | {s['dev_spur_ap']:.3f} | "
              f"{s['dev_spur_auc']:.3f} | {100 * s['top1']:.2f} | {100 * diff.mean().item():+.2f} ({100 * se:.2f}) | "
              f"{100 * s['top1_dev_classes']:.1f} | {s['unseen_spur_hit']:.3f} | {s['unseen_spur_ap']:.3f} | "
              f"{100 * s['top1_unseen_classes']:.1f} |")
    cond1 = lambda s: s["dev_spur_hit"] <= 0.5 * base["dev_spur_hit"] and s["dev_spur_ap"] >= base["dev_spur_ap"] + 0.10  # noqa: E731
    cond2 = lambda s: s["top1"] >= base["top1"] - 0.005 and s["top1_dev_classes"] >= base["top1_dev_classes"] - 0.03  # noqa: E731
    print("\npass bar: 1 = hit <= %.3f and AP >= %.3f; 2 = top-1 >= %.4f and dev-class top-1 >= %.4f" % (
        0.5 * base["dev_spur_hit"], base["dev_spur_ap"] + 0.10, base["top1"] - 0.005, base["top1_dev_classes"] - 0.03))
    for a, s in S.items():
        line = f"{a}: cond1 {cond1(s)}  cond2 {cond2(s)}"
        if a.startswith("agent_"):
            r = S.get("random_same_" + a.split("_")[1])
            c3 = r is not None and not cond1(r)
            line += f"  cond3 {c3}  PASS {cond1(s) and cond2(s) and c3}"
        print(line)
    # per-class view for agents: classes whose held-out AP improved / worsened
    for a in [x for x in S if x.startswith("agent_") or x in ("proj", "ft_l4")]:
        P0 = json.loads((E / "none.json").read_text())["per_class"]["dev"]
        P1 = json.loads((E / f"{a}.json").read_text())["per_class"]["dev"]
        dap = [P1[k]["ap"] - P0[k]["ap"] for k in P0]
        dct = [P1[k]["clean_top1"] - P0[k]["clean_top1"] for k in P0]
        print(f"{a}: AP up in {sum(x > 0.01 for x in dap)}/50 classes, down in {sum(x < -0.01 for x in dap)}; "
              f"clean top-1 of class k down in {sum(x < 0 for x in dct)}/50, up in {sum(x > 0 for x in dct)}")


if __name__ == "__main__":
    main()
