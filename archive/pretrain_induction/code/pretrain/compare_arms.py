"""Compare continued-training arms from one agent session against the unedited control.

Usage: .venv/Scripts/python.exe pretrain/compare_arms.py runs/pretrain_agent/<session> [--out plot.png]
Reads <session>/arm_*/log.jsonl. arm_none is the control.
Reports per arm: mean val loss over the resumed span, val loss at fixed steps, and "steps saved":
how many steps later the control first reaches the arm's val loss (positive = arm is ahead).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def probes(run):
    P = [json.loads(l) for l in open(os.path.join(run, "log.jsonl")) if '"probe"' in l]
    s = np.array([p["step"] for p in P])
    v = np.array([p["probe"]["val_loss"] for p in P])
    c = np.array([p["probe"]["rep1"] - p["probe"]["rep2"] for p in P])
    return s, v, c


def smooth(y, k=9):
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def steps_saved(s, v, s0, v0):
    """For each arm probe, steps until the control reaches that loss (both curves smoothed over 9 probes,
    control made monotone). Noisy where the curve is flat: late in training 0.01 nats ~ hundreds of steps."""
    v = smooth(v)
    env = np.minimum.accumulate(smooth(v0))
    out = []
    for st, val in zip(s, v):
        idx = np.flatnonzero(env <= val)
        out.append((s0[idx[0]] - st) if len(idx) else np.nan)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = {os.path.basename(p)[4:]: p for p in sorted(glob.glob(os.path.join(a.session, "arm_*"))) if os.path.isdir(p)}
    s0, v0, c0 = probes(arms["none"])
    checks = [1250, 1500, 2000, 3000, 5000, 10000, 15000]
    print(f"{'arm':10s} {'mean val':>9s} " + " ".join(f"{k:>7d}" for k in checks) + "   steps saved @ " + " ".join(str(k) for k in checks))
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    for i, (name, path) in enumerate(arms.items()):
        s, v, c = probes(path)
        n = min(len(s), len(s0))
        assert (s[:n] == s0[:n]).all()
        sav = steps_saved(s, v, s0, v0)
        at = {k: v[s == k][0] for k in checks if (s == k).any()}
        sv = {k: sav[s == k][0] for k in checks if (s == k).any()}
        print(f"{name:10s} {v.mean():9.4f} " + " ".join(f"{at.get(k, np.nan):7.3f}" for k in checks)
              + "   " + " ".join(f"{sv.get(k, np.nan):5.0f}" for k in checks))
        col = {"none": "k", "agent": "C3"}.get(name, f"C{i}")
        ax[0].plot(s, v, color=col, label=name, lw=1.3)
        ax[1].plot(s[:n], v[:n] - v0[:n], color=col, label=f"{name} - none", lw=1.2)
        ax[2].plot(s, c, color=col, label=name, lw=1.2)
        if name != "none":
            ax[3].plot(s, sav, color=col, label=name, lw=1.2)
    ax[0].set_title("val loss (held-out)"); ax[0].set_ylim(top=3.2)
    ax[1].set_title("val loss minus control (negative = better)"); ax[1].axhline(0, color="k", lw=0.6)
    ax[2].set_title("copy benefit (induction)")
    ax[3].set_title("steps saved: how much later the control reaches this loss"); ax[3].axhline(0, color="k", lw=0.6)
    for x in ax:
        x.set_xscale("log"); x.grid(alpha=0.3); x.set_xlabel("step (log)"); x.legend(fontsize=8)
    fig.tight_layout()
    out = a.out or os.path.join(a.session, "compare.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
