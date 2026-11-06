"""Diagnostic held-out trade-off plot (not used for any selection).
  python vision_spurious/plot_tradeoff.py                                  -> tradeoff.png (A1 curves)
  python vision_spurious/plot_tradeoff.py curves_in tradeoff_imagenet.png  -> A2 curves (ImageNet-train data)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

E = C.RUNS / "eval"
CU = C.RUNS / (sys.argv[1] if len(sys.argv) > 1 else "curves")
PNG = sys.argv[2] if len(sys.argv) > 2 else "tradeoff.png"
VARIANTS = ["llr", "llr_rows", "llr_gentle", "ft", "ft_gentle", "llr_rows_in", "llr_gentle_in", "ft_in", "ft_gentle_in",
            "dfr_lam0", "dfr_lam0.01", "dfr_lam0.1", "dfr_lam1"]
COL = dict(llr="#1f77b4", llr_rows="#17becf", llr_gentle="#9467bd", ft="#d62728", ft_gentle="#ff7f0e",
           llr_rows_in="#17becf", llr_gentle_in="#9467bd", ft_in="#d62728", ft_gentle_in="#ff7f0e",
           **{"dfr_lam0": "#8c564b", "dfr_lam0.01": "#a0522d", "dfr_lam0.1": "#bc8f8f", "dfr_lam1": "#e377c2"})
SCRIPT = [("script_s2_pool", "P"), ("script_s2_dev3", "p"), ("script_s1_pool", "h"), ("script_s1_dev3", "H")]


def S(name):
    return json.loads((E / f"{name}.json").read_text())["summary"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, key, lab in [(axes[0], "top1_dev_classes", "top-1 on the 50 dev classes (held-out, 350 photos)"),
                         (axes[1], "top1", "overall top-1 (held-out ImageNetV2, 7,000 photos)")]:
        for v in VARIANTS:
            p = CU / f"{v}.json"
            if not p.exists():
                continue
            J = json.loads(p.read_text())
            cur = J["heldout_curve"]
            ax.plot([r["dev_spur_hit"] for r in cur], [100 * r[key] for r in cur], "-", color=COL[v], lw=1.3,
                    alpha=0.8, label=f"{v} (trajectory)")
            for lbl, sel in J["selected"].items():
                h = sel["heldout"]
                ax.plot(h["dev_spur_hit"], 100 * h[key], "s" if lbl in ("cv", "prac") else "D", color=COL[v], ms=6,
                        mec="k", mew=0.5)
        base = S("none")
        ax.plot(base["dev_spur_hit"], 100 * base[key], "ko", ms=8, label="no fix")
        for i in (1, 2, 3):
            a = S(f"agent_s{i}")
            ax.plot(a["dev_spur_hit"], 100 * a[key], "*", color="#2ca02c", ms=16, mec="k", mew=0.6,
                    label="Opus edits" if i == 1 else None)
            ax.annotate(f"s{i}", (a["dev_spur_hit"], 100 * a[key]), xytext=(6, 4), textcoords="offset points")
            for mode, mk in (("same", "x"), ("count", "+")):
                r = S(f"random_{mode}_s{i}")
                ax.plot(r["dev_spur_hit"], 100 * r[key], mk, color="grey", ms=7,
                        label=f"random ({mode})" if i == 1 else None)
        for n, mk in (("proj", "^"), ("spufix_like", "v")):
            r = S(n)
            ax.plot(r["dev_spur_hit"], 100 * r[key], mk, color="k", ms=7, mfc="none", label=n)
        if CU.name == "curves_in":
            for n, mk in SCRIPT:
                if (E / f"{n}.json").exists():
                    r = S(n)
                    ax.plot(r["dev_spur_hit"], 100 * r[key], mk, color="#006400", ms=9, mfc="none", mew=1.4, label=n)
        b = 100 * base[key]
        ax.axhline(b - (3 if key == "top1_dev_classes" else 0.5), color="k", ls=":", lw=0.8)
        ax.axvline(0.5 * base["dev_spur_hit"], color="k", ls=":", lw=0.8)
        ax.set_xlabel("held-out spurious hit rate, dev classes (lower = better)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Held-out trade-off: training trajectories (lines; diamonds = matched-hit checkpoints, squares = "
                 "dev-only stopping rule (CV / practitioner)) vs Opus edits.\nDotted: pass-bar limits. Diagnostic only, not used for "
                 "selection.", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(C.RUNS / PNG, dpi=130)
    print("wrote", C.RUNS / PNG)


if __name__ == "__main__":
    main()
