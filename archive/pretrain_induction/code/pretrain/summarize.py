"""Plateau summary per run: when induction appears, how sharply, and what it is worth.

copy benefit = loss(1st copy) - loss(2nd copy) on repeated real text (rep1 - rep2).
  onset  = first probe where the (5-probe smoothed) copy benefit reaches 0.25
  t50    = first probe where it reaches 50% of its final value
  t90    = first probe where it reaches 90% of its final value
icl = loss(late positions) - loss(early positions) on val; more negative = more use of context.

Usage: .venv/Scripts/python.exe pretrain/summarize.py runs/pretrain_fw/* runs/pretrain_code/*
"""
import json
import os
import sys

import numpy as np


def first(steps, cond):
    idx = np.flatnonzero(cond)
    return int(steps[idx[0]]) if len(idx) else None


def summarize(run):
    pr = [json.loads(l) for l in open(os.path.join(run, "log.jsonl")) if '"probe"' in l]
    if not pr:
        return None
    s = np.array([p["step"] for p in pr])
    g = lambda k: np.array([p["probe"][k] for p in pr])
    cb = g("rep1") - g("rep2")
    cbs = np.convolve(np.pad(cb, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    final = cbs[-5:].mean()
    ind = np.array([max(max(l) for l in p["probe"]["ind"]) for p in pr])
    val = g("val_loss")
    at = lambda step: float(val[np.argmin(np.abs(s - step))])
    cfg = json.load(open(os.path.join(run, "config.json")))
    return {
        "run": f"{os.path.basename(os.path.dirname(os.path.normpath(run))).replace('pretrain_', '')}/{os.path.basename(os.path.normpath(run))}",
        "params_M": round(cfg["params"] / 1e6, 1),
        "last_step": int(s[-1]),
        "onset": first(s, cbs >= 0.25),
        "t50": first(s, cbs >= 0.5 * final) if final > 0.25 else None,
        "t90": first(s, cbs >= 0.9 * final) if final > 0.25 else None,
        "copy_final": round(float(final), 2),
        "ind_max": round(float(ind[-5:].mean()), 2),
        "icl_final": round(float(g("icl")[-5:].mean()), 2),
        "val@1k": round(at(1000), 3), "val@3k": round(at(3000), 3), "val_final": round(float(val[-1]), 3),
    }


def main():
    rows = [r for r in (summarize(p) for p in sys.argv[1:] if os.path.isfile(os.path.join(p, "log.jsonl"))) if r]
    keys = list(rows[0].keys())
    w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(w[k]) for k in keys))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w[k]) for k in keys))


if __name__ == "__main__":
    main()
