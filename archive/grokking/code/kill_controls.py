"""Follow-up checks on the Opus 'kill neurons' edits.

1. Dead-neuron check: after continuing from the Opus edit, how many killed neurons ever fire again?
2. Random-kill control: kill the same number of *random* neurons with the same bias values.
Results go to runs/seedN/kill_controls.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"<project>")
sys.path.insert(0, str(ROOT))
from experiment import continue_arm  # noqa: E402
from grok import Run, set_weights  # noqa: E402

for seed in map(int, sys.argv[1:]):
    out = ROOT / "runs" / f"seed{seed}"
    ckpt = out / "edit_point.pt"
    orig = dict(np.load(out / "agent" / "weights.npz"))
    opus = dict(np.load(out / "agent" / "submission.npz"))
    killed = np.where((opus["fc1.bias"] != orig["fc1.bias"]) & (opus["fc1.bias"] < -1))[0]
    res = dict(seed=seed, killed=killed.tolist())

    run = Run.load(ckpt)
    set_weights(run.model, opus)
    alive_after = {}
    for chunk in range(6):
        run.train(10_000)
        with torch.no_grad():
            _, h = run.model(run.xtr, return_hidden=True)
        active = (h > 0).any(0).cpu().numpy()
        alive_after[(chunk + 1) * 10_000] = dict(killed_firing=int(active[killed].sum()),
                                                total_firing=int(active.sum()),
                                                test_acc=round(run.evaluate()["test_acc"], 4))
    res["opus_dead_check"] = alive_after
    print(f"seed {seed}: killed {len(killed)} | {alive_after}", flush=True)

    rng = np.random.default_rng(2000 + seed)
    res["random_kill"] = {}
    for i in range(3):
        w = {k: v.copy() for k, v in orig.items()}
        pick = rng.choice(orig["fc1.bias"].size, len(killed), replace=False)
        w["fc1.bias"][pick] = rng.permutation(opus["fc1.bias"][killed])
        r = continue_arm(ckpt, w)
        best = max(m["test_acc"] for m in r["history"])
        res["random_kill"][f"rk{i}"] = dict(right_after=r["right_after"], steps_to=r["steps_to"], best_test=best)
        print(f"seed {seed} random-kill{i}: right after train {r['right_after']['train_acc']:.1%} "
              f"test {r['right_after']['test_acc']:.1%} | steps_to {r['steps_to']} | best test {best:.3f}", flush=True)
    (out / "kill_controls.json").write_text(json.dumps(res, indent=1))
