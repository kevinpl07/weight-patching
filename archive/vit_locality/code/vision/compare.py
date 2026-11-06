"""Compare the continued-training arms of the ViT locality experiment across seeds.

Usage: .venv/Scripts/python.exe vision/compare.py runs/vision_vit/s*_step1000 [--extra name=dir ...] [--out plot.png]
Each session dir holds arm_<name>/log.jsonl (arm_none is the control). --extra adds whole from-scratch runs
(e.g. mimetic0=runs/vision_vit/mimetic0/s{seed}; "{seed}" is replaced per session) as extra arms.

Reports per seed and arm, then the mean over seeds:
  final    test accuracy at the last step
  last5    mean test accuracy over the last 5 probes (less noisy)
  span     mean test accuracy over all probes from the edit step to the end
  t@X      first step at which test accuracy >= X (head start; "-" = never)
  d0..dL   mean attention distance per layer (mean over heads) at selected steps
and the per-seed difference to arm_none for final / last5 / span.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEVELS = (0.60, 0.70, 0.75, 0.80)


def probes(run):
    P = [json.loads(l) for l in open(os.path.join(run, "log.jsonl")) if '"probe"' in l]
    d = {}
    for p in P:  # later lines win (a resumed run appends)
        d[p["step"]] = p["probe"]
    s = np.array(sorted(d))
    return s, [d[k] for k in s]


def summary(run, edit_step):
    s, P = probes(run)
    acc = np.array([p["test_acc"] for p in P])
    m = s >= edit_step
    out = dict(steps=s, acc=acc, final=acc[-1], last5=acc[-5:].mean(), span=acc[m].mean(), last_step=s[-1],
               dist=np.array([[np.mean(r) for r in p["dist"]] for p in P]),
               local=np.array([[np.mean(r) for r in p["local"]] for p in P]))
    for L in LEVELS:
        hit = np.flatnonzero(acc >= L)
        out[f"t@{L:.2f}"] = int(s[hit[0]]) if len(hit) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--extra", nargs="*", default=[], help="name=dir with {seed}")
    ap.add_argument("--out", default="runs/vision_vit/compare.png")
    a = ap.parse_args()
    sessions = sorted({p for s in a.sessions for p in glob.glob(s) if os.path.isdir(p)})
    table = {}  # arm -> list of (seed, summary)
    for sess in sessions:
        meta = json.load(open(os.path.join(sess, "session.json")))
        seed = int(re.search(r"s(\d+)_step", os.path.basename(sess)).group(1))
        arms = {os.path.basename(p)[4:]: p for p in sorted(glob.glob(os.path.join(sess, "arm_*"))) if os.path.isdir(p)}
        for e in a.extra:
            name, pat = e.split("=", 1)
            p = pat.replace("{seed}", str(seed))
            if os.path.isdir(p):
                arms[name] = p
        for name, path in arms.items():
            try:
                table.setdefault(name, []).append((seed, summary(path, meta["step"])))
            except (FileNotFoundError, IndexError):
                pass
        edit_step, total = meta["step"], meta["total_steps"]
    none = {sd: r for sd, r in table.get("none", [])}
    hdr = f"{'arm':12s} {'seed':>4s} {'final':>7s} {'last5':>7s} {'span':>7s} " + " ".join(f"t@{L:.2f}" for L in LEVELS) \
        + "   d_final(layer means)          dF-none  d5-none  dS-none"
    print(hdr)
    agg = {}
    for name, rows in table.items():
        for seed, r in sorted(rows, key=lambda t: t[0]):
            done = r["last_step"] >= total
            diffs = ""
            if name != "none" and seed in none and done and none[seed]["last_step"] >= total:
                n = none[seed]
                diffs = f"{100*(r['final']-n['final']):+7.2f}  {100*(r['last5']-n['last5']):+7.2f}  {100*(r['span']-n['span']):+7.2f}"
                agg.setdefault(name, []).append((r["final"] - n["final"], r["last5"] - n["last5"], r["span"] - n["span"]))
            ts = " ".join(f"{(str(r[f't@{L:.2f}']) if r[f't@{L:.2f}'] is not None else '-'):>6s}" for L in LEVELS)
            dd = " ".join(f"{v:.2f}" for v in r["dist"][-1])
            print(f"{name:12s} {seed:4d} {100*r['final']:7.2f} {100*r['last5']:7.2f} {100*r['span']:7.2f} {ts}   "
                  f"{dd}  {diffs}{'' if done else '  (running, step %d)' % r['last_step']}")
    print("\nmean over seeds (finished runs):")
    for name, rows in table.items():
        rows = [r for _, r in rows if r["last_step"] >= total]
        if not rows:
            continue
        f = np.array([r["final"] for r in rows]); l5 = np.array([r["last5"] for r in rows])
        sp = np.array([r["span"] for r in rows])
        line = (f"{name:12s} n={len(rows)}  final {100*f.mean():.2f} ± {100*f.std(ddof=1) if len(f)>1 else 0:.2f}  "
                f"last5 {100*l5.mean():.2f}  span {100*sp.mean():.2f}")
        if name in agg:
            d = np.array(agg[name]) * 100
            line += (f"   vs none: final {d[:,0].mean():+.2f} (per seed {', '.join(f'{v:+.2f}' for v in d[:,0])}), "
                     f"last5 {d[:,1].mean():+.2f}, span {d[:,2].mean():+.2f}")
        print(line)

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    cols = {"none": "k", "agent": "C3", "mimetic": "C2", "mimetic0": "C4"}
    for i, (name, rows) in enumerate(table.items()):
        c = cols.get(name, f"C{i % 10}")
        for j, (seed, r) in enumerate(rows):
            lab = name if j == 0 else None
            ax[0].plot(r["steps"], 100 * r["acc"], color=c, lw=1, alpha=0.8, label=lab)
            if name != "none" and seed in none:
                n = none[seed]
                k = min(len(n["steps"]), len(r["steps"]))
                m = r["steps"][:k] >= edit_step
                if not (n["steps"][:k] == r["steps"][:k]).all():
                    continue
                ax[1].plot(r["steps"][:k][m], 100 * (r["acc"][:k] - n["acc"][:k])[m], color=c, lw=1, alpha=0.8, label=lab)
            ax[2].plot(r["steps"], r["dist"].mean(1), color=c, lw=1, alpha=0.8, label=lab)
            ax[3].plot(r["steps"], r["dist"][:, 0], color=c, lw=1, alpha=0.8, label=lab)
    ax[0].set_title("test accuracy (%)"); ax[0].set_ylim(30, 95)
    ax[1].set_title("test accuracy minus arm_none, same seed (points)"); ax[1].axhline(0, color="k", lw=0.6)
    ax[2].set_title("mean attention distance (all layers)"); ax[3].set_title("mean attention distance, layer 0")
    for x in ax:
        x.grid(alpha=0.3); x.set_xlabel("step"); x.legend(fontsize=8); x.axvline(edit_step, color="gray", lw=0.6, ls="--")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
