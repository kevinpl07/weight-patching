"""Capacity check: do narrower networks, trained from scratch with the same settings, still grok?

Hidden sizes match the working-neuron counts the Opus loop ended with (54, 60, 84).
Each run trains up to 62,300 steps (edit point + 60k), stopping once held-out reaches 95%.
Results go to runs/capacity_hN.json.

    python capacity_check.py 54 60 84
"""
import json
import sys
from dataclasses import replace

from grok import Run, first_step  # first: sets CUBLAS_WORKSPACE_CONFIG
from experiment import CFG, GROK_ACC, RUNS

SIZES = [int(a) for a in sys.argv[1:]] or [54, 60, 84]
SEEDS = [0, 1, 2]
MAX_STEPS = 62_300

out = {}
for hidden in SIZES:
    for seed in SEEDS:
        run = Run(replace(CFG, hidden=hidden), seed)
        run.train(MAX_STEPS, stop=lambda m: m["test_acc"] >= GROK_ACC)
        h = run.history
        res = dict(hidden=hidden, seed=seed, memorized=first_step(h, "train_acc", 1.0),
                   test50=first_step(h, "test_acc", 0.5), test95=first_step(h, "test_acc", GROK_ACC),
                   best_test=max(m["test_acc"] for m in h), last_step=h[-1]["step"])
        out[f"h{hidden}_s{seed}"] = res
        print(res, flush=True)
(RUNS / f"capacity_h{'_'.join(map(str, SIZES))}.json").write_text(json.dumps(out, indent=1))
