"""Plot loss and plateau probes for one or more pretraining runs.

Usage: .venv/Scripts/python.exe pretrain/plot.py runs/pretrain/base_s0 [more runs...] --out plot.png [--log-x]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(run):
    steps, loss, probes = [], [], []
    for line in open(os.path.join(run, "log.jsonl")):
        r = json.loads(line)
        if "loss" in r:
            steps.append(r["step"]); loss.append(r["loss"])
        else:
            probes.append(r)
    return np.array(steps), np.array(loss), probes


def smooth(y, k=9):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="plot.png")
    ap.add_argument("--log-x", action="store_true")
    ap.add_argument("--max-step", type=int, default=None)
    a = ap.parse_args()

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()
    for i, run in enumerate(a.runs):
        name = os.path.basename(os.path.normpath(run))
        c = f"C{i % 10}"
        s, l, pr = load(run)
        if a.max_step:
            m = s <= a.max_step; s, l = s[m], l[m]
            pr = [p for p in pr if p["step"] <= a.max_step]
        ps = np.array([p["step"] for p in pr])
        P = lambda k: np.array([p["probe"][k] for p in pr])
        ax[0].plot(s, l, color=c, alpha=0.15, lw=0.6)
        ax[0].plot(s, smooth(l), color=c, lw=1.3, label=name)
        ax[1].plot(ps, P("val_loss"), color=c, label=name)
        ax[2].plot(ps, P("rep1"), color=c, ls=":", lw=1)
        ax[2].plot(ps, P("rep2"), color=c, label=name)
        ax[3].plot(ps, P("rep1") - P("rep2"), color=c, label=f"{name} story text")
        if "rrep1" in pr[0]["probe"]:
            ax[3].plot(ps, P("rrep1") - P("rrep2"), color=c, ls=":", lw=0.8, label=f"{name} random tokens")
        ind = np.array([[max(layer) for layer in p["probe"]["ind"]] for p in pr])
        prev = np.array([[max(layer) for layer in p["probe"]["prev"]] for p in pr])
        styles = ["-", "--", "-.", ":"]
        for L in range(ind.shape[1]):
            ax[4].plot(ps, ind[:, L], color=c, lw=1.2, ls=styles[L % 4], label=f"{name} L{L}")
            ax[5].plot(ps, prev[:, L], color=c, lw=1.2, ls=styles[L % 4], label=f"{name} L{L}")

    titles = ["train loss (raw + smoothed)", "val loss (whole window inside one story)",
              "repeated story text: loss on 2nd copy (solid) vs 1st (dotted)",
              "copy benefit = loss(1st copy) - loss(2nd copy)  [induction]",
              "max induction attention score per layer", "max previous-token attention score per layer"]
    for k, t in enumerate(titles):
        ax[k].set_title(t, fontsize=10)
        ax[k].grid(alpha=0.3)
        ax[k].set_xlabel("step")
        if a.log_x:
            ax[k].set_xscale("symlog", linthresh=10)
    ax[0].set_ylim(top=min(9, ax[0].get_ylim()[1]))
    for k in (0, 3, 4, 5):
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
