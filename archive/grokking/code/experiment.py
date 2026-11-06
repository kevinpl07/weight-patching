"""Pilot: does a direct Opus weight edit in the middle of training speed up grokking?

    python experiment.py tune --seeds 0 1 2      # plain runs, prints when each one memorizes / groks
    python experiment.py prepare --seed 0        # train to the edit point, build the agent session
    # run the Opus agent on runs/seed0/agent (prompt: runs/seed0/agent/PROMPT.md)
    python experiment.py forks --seed 0          # continue: unedited vs Opus edit vs random edits
    python experiment.py report

Edits are never made by an optimizer: the agent writes values through edit_tools.py.
Every arm resumes with the same AdamW state (moments are kept, not reset) and the same steps.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from grok import Config, Run, first_step, get_weights, set_weights

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
CFG = Config()
EDIT_AFTER_MEMORIZED = 1000  # steps of plateau between 100% train accuracy and the edit
BUDGET_FRAC = 0.02           # max fraction of weight entries the agent may change
GROK_ACC = 0.95
MAX_CONTINUE = 60_000
N_RANDOM = 3


def milestones(history, after=0):
    return dict(memorized=first_step(history, "train_acc", 1.0, after),
                test50=first_step(history, "test_acc", 0.5, after),
                test95=first_step(history, "test_acc", GROK_ACC, after))


def cmd_tune(args):
    for seed in args.seeds:
        run = Run(CFG, seed)
        run.train(args.steps, stop=lambda m: m["test_acc"] >= GROK_ACC)
        print(f"seed {seed}: {milestones(run.history)}  final {run.history[-1]}")


def cmd_prepare(args):
    out = RUNS / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    run = Run(CFG, args.seed)
    m = run.train(MAX_CONTINUE, stop=lambda m: m["train_acc"] >= 1.0)
    run.train(EDIT_AFTER_MEMORIZED)
    m = run.evaluate()
    print(f"seed {args.seed}: memorized at {first_step(run.history, 'train_acc', 1.0)}, edit point {m}")
    run.save(out / "edit_point.pt")

    session = out / "agent"
    if session.exists():
        shutil.rmtree(session)
    (session / "scratch").mkdir(parents=True)
    weights = get_weights(run.model)
    np.savez(session / "weights.npz", **weights)
    np.savez(session / "working.npz", **weights)
    labels = (run.train_pairs[:, 0] + run.train_pairs[:, 1]) % CFG.p
    np.save(session / "train_pairs.npy", np.column_stack([run.train_pairs, labels]))
    total = sum(w.size for w in weights.values())
    meta = dict(cfg=CFG.__dict__, seed=args.seed, step=run.step, total_weights=int(total),
                budget=int(BUDGET_FRAC * total), state=m,
                history=[h for h in run.history if h["step"] % 500 == 0])
    (session / "meta.json").write_text(json.dumps(meta, indent=1))
    prompt = (ROOT / "agent_prompt.md").read_text(encoding="utf-8")
    for k, v in dict(SESSION=session.resolve(), ROOT=ROOT.resolve(), BUDGET=meta["budget"], TOTAL=total,
                     STEP=run.step, TRAIN_ACC=f"{m['train_acc']:.1%}", TEST_ACC=f"{m['test_acc']:.1%}",
                     N_TRAIN=len(run.train_pairs), **{k.upper(): v for k, v in CFG.__dict__.items()}).items():
        prompt = prompt.replace("{" + k + "}", str(v))
    (session / "PROMPT.md").write_text(prompt, encoding="utf-8")
    print(f"agent session ready: {session}  (budget {meta['budget']} of {total} weights)")


def random_edit(orig, edited, rng):
    """Same number of changed entries and same delta L2 norm per tensor, random positions and directions."""
    out = {}
    for k, w in orig.items():
        d = edited[k] - w
        n, norm = int((d != 0).sum()), float(np.linalg.norm(d))
        new = w.copy()
        if n:
            pos = rng.choice(w.size, n, replace=False)
            delta = rng.standard_normal(n)
            new.flat[pos] += (delta * norm / np.linalg.norm(delta)).astype(w.dtype)
        out[k] = new
    return out


def continue_arm(ckpt, weights):
    run = Run.load(ckpt)
    start = run.step
    if weights is not None:
        set_weights(run.model, weights)
    right_after = run.evaluate()
    run.history = [right_after]
    run.train(MAX_CONTINUE, stop=lambda m: m["test_acc"] >= GROK_ACC)
    ms = {k: (v - start if v is not None else None) for k, v in milestones(run.history, start).items()}
    return dict(right_after=right_after, steps_to=ms, history=run.history)


def cmd_forks(args):
    out = RUNS / f"seed{args.seed}"
    ckpt = out / "edit_point.pt"
    session = out / "agent"
    orig = dict(np.load(session / "weights.npz"))
    opus = dict(np.load(session / "submission.npz"))
    rng = np.random.default_rng(1000 + args.seed)
    arms = {"unedited": None, "opus": opus}
    arms.update({f"random{i}": random_edit(orig, opus, rng) for i in range(N_RANDOM)})
    results = dict(seed=args.seed, edit=json.loads((session / "submission.json").read_text()), arms={})
    for name, w in arms.items():
        r = continue_arm(ckpt, w)
        results["arms"][name] = r
        ra = r["right_after"]
        print(f"seed {args.seed} {name:9s} right after: train {ra['train_acc']:.1%} test {ra['test_acc']:.1%}"
              f" | steps to test50 {r['steps_to']['test50']}, test95 {r['steps_to']['test95']}")
    (out / "results.json").write_text(json.dumps(results, indent=1))


def cmd_report(args):
    rows = []
    for f in sorted(RUNS.glob("seed*/results.json")):
        r = json.loads(f.read_text())
        a = r["arms"]
        rnd = [a[k]["steps_to"]["test95"] for k in a if k.startswith("random")]
        rows.append((r["seed"], a["unedited"]["steps_to"]["test95"], a["opus"]["steps_to"]["test95"], rnd,
                     r["edit"]["changed"], a["opus"]["right_after"]["train_acc"], a["opus"]["right_after"]["test_acc"]))
    print("seed | unedited | opus | random (x3) | changed | opus right after (train/test)")
    for s, u, o, rnd, ch, tr, te in rows:
        print(f"{s:4d} | {u} | {o} | {rnd} | {ch} | {tr:.1%} / {te:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--steps", type=int, default=60_000)
    for name in ("prepare", "forks"):
        sub.add_parser(name).add_argument("--seed", type=int, required=True)
    sub.add_parser("report")
    args = ap.parse_args()
    dict(tune=cmd_tune, prepare=cmd_prepare, forks=cmd_forks, report=cmd_report)[args.cmd](args)
