"""Pilot 2: repeated Opus edits during training (see NEXT_EXPERIMENT.md).

    python loop.py start --seed 0      # round 1 = pilot submission, train 4k, build round 2 session
    # agent turn: give runs/seed0/loop/roundN/agent/PROMPT.md to the model as its task
    python loop.py next --seed 0       # apply this round's submission, train 4k, build the next round
                                       # (after round 4: train to the 60k cap, write loop/results.json)
    python loop.py revival --seed 0    # control: Opus round 1, then scripted revival of dead neurons at +4k
    python loop.py random --seed 0     # control: random edits matched to each Opus round
    python loop.py report

    python loop.py --v2 start --seed 0   # pilot 3: fresh round 1 at the edit point, v2 prompt and tools
    python loop.py --v2 next --seed 0    # (v2 sessions get the optimizer view, optimizer resets, trial
    python loop.py --v2 random --seed 0  #  runs and predictions; results go to runs/seedN/loop2/)

Edits are never made by an optimizer. Every arm keeps its AdamW moments across edits
(except the revived entries in the revival arm, whose moments are reset).
"""
import argparse
import json
import math
import shutil
from pathlib import Path

from grok import Run, first_step, get_weights, set_weights  # first: sets CUBLAS_WORKSPACE_CONFIG

import numpy as np
import torch
from experiment import BUDGET_FRAC, CFG, GROK_ACC, MAX_CONTINUE, RUNS, ROOT, milestones, random_edit

N_ROUNDS = 4
ROUND_STEPS = 4000
N_RANDOM_DRAWS = 2
V2 = False  # set by --v2: pilot 3 (loop2/, loop_prompt_v2.md, optimizer view, resets, trials, predictions)


def seed_dir(seed):
    return RUNS / f"seed{seed}"


def loop_dir(seed):
    return seed_dir(seed) / ("loop2" if V2 else "loop")


def round_session(seed, r):
    """Session folder holding round r's weights.npz and submission.npz (in v1, round 1 is the pilot's)."""
    return seed_dir(seed) / "agent" if r == 1 and not V2 else loop_dir(seed) / f"round{r}" / "agent"


def load_state(seed):
    return json.loads((loop_dir(seed) / "state.json").read_text())


def save_state(seed, state):
    (loop_dir(seed) / "state.json").write_text(json.dumps(state, indent=1))


@torch.no_grad()
def firing(run):
    """Bool per hidden neuron: does it fire on at least one training pair?"""
    _, h = run.model(run.xtr, return_hidden=True)
    return (h > 0).any(0).cpu().numpy()


def apply_edit(run, weights, label):
    """Set weights at the current step and log the evaluation right after, marked with `edit`."""
    set_weights(run.model, weights)
    m = run.evaluate()
    m["edit"] = label
    run.history.append(m)
    return m


def reset_moments(run, masks):
    """Zero the AdamW moments of the entries marked in masks ({tensor name: bool array})."""
    with torch.no_grad():
        for k, p in run.model.named_parameters():
            mask = torch.as_tensor(masks[k], device=p.device)
            for key in ("exp_avg", "exp_avg_sq"):
                run.opt.state[p][key][mask] = 0
    return int(sum(m.sum() for m in masks.values()))


def plain(history):
    return [h for h in history if "edit" not in h]


def summarize(history, start):
    after = [h for h in history if h["step"] >= start]
    ms = {k: (v - start if v is not None else None) for k, v in milestones(history, start).items()}
    return dict(steps_to=ms, best_test=max(h["test_acc"] for h in after),
                final=after[-1], best_test_step=max(after, key=lambda h: h["test_acc"])["step"] - start)


def curve_table(history, since, until):
    """Markdown table of the curve from an edit at `since` (right after it) to `until`, every 500 steps."""
    rows = ["| steps since edit | step | train loss | train acc | held-out acc |", "|---|---|---|---|---|"]
    after = next(h for h in history if h["step"] == since and "edit" in h)
    pts = [("0 (right after edit)", after)] + [
        (str(h["step"] - since), h) for h in plain(history)
        if since < h["step"] <= until and (h["step"] - since) % 500 == 0]
    for label, h in pts:
        rows.append(f"| {label} | {h['step']} | {h['train_loss']:.4f} | {h['train_acc']:.1%} | {h['test_acc']:.1%} |")
    return "\n".join(rows)


def write_history_md(path, run, state):
    memorized = first_step(plain(run.history), "train_acc", 1.0)
    parts = [f"# Edit history\n\nTraining started at step 0. Training accuracy first reached 100% at step {memorized}. "
             f"Each round below was a separate session with the same instructions; training ran "
             f"{ROUND_STEPS:,} steps after each edit, with the same AdamW optimizer state carried through.\n"]
    if not state["rounds"]:
        parts.append("No earlier rounds: this is the first edit.\n")
    for r, info in enumerate(state["rounds"], 1):
        sub = json.loads((round_session(state["seed"], r) / "submission.json").read_text())
        step = info["step"]
        before = next(h for h in plain(run.history) if h["step"] == step)
        per = ", ".join(f"{k} {v}" for k, v in sub["per_tensor"].items())
        pred = ""
        if sub.get("predict_next") is not None:
            later = next((h for h in plain(run.history) if h["step"] == step + ROUND_STEPS), None)
            actual = f"{later['test_acc']:.1%}" if later else "not reached yet"
            resets = sum(sub.get("optimizer_resets", {}).values())
            pred = (f"- Optimizer-state resets: {resets} entries\n"
                    f"- Predicted held-out acc {ROUND_STEPS:,} steps after the edit: {sub['predict_next']:.1%}; "
                    f"actual: {actual}\n"
                    f"- Predicted held-out acc at the end of the run (step {state['start'] + MAX_CONTINUE:,}): "
                    f"{sub['predict_end']:.1%}\n")
        parts.append(f"## Round {r}: edit at step {step}\n\n"
                     f"- Changed entries: {sub['changed']} ({per})\n"
                     f"- Submission note: \"{sub['note']}\"\n{pred}"
                     f"- Right before the edit: train acc {before['train_acc']:.1%}, held-out acc {before['test_acc']:.1%}, "
                     f"train loss {before['train_loss']:.4f}\n\n"
                     f"Curve since this edit:\n\n{curve_table(run.history, step, step + ROUND_STEPS)}\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def build_session(seed, r, run, state):
    """Round r's agent session from the paused run (prompt, weights, meta, history.md)."""
    session = round_session(seed, r)
    if session.exists():
        shutil.rmtree(session)
    (session / "scratch").mkdir(parents=True)
    weights = get_weights(run.model)
    np.savez(session / "weights.npz", **weights)
    np.savez(session / "working.npz", **weights)
    labels = (run.train_pairs[:, 0] + run.train_pairs[:, 1]) % CFG.p
    np.save(session / "train_pairs.npy", np.column_stack([run.train_pairs, labels]))
    m = run.evaluate()
    total = sum(w.size for w in weights.values())
    meta = dict(cfg=CFG.__dict__, seed=seed, step=run.step, total_weights=int(total),
                budget=int(BUDGET_FRAC * total), state=m, round=r,
                history=[h for h in plain(run.history) if h["step"] % 500 == 0 and h["step"] < run.step])
    if V2:
        meta["checkpoint"] = str((loop_dir(seed) / f"round{r}" / "checkpoint.pt").resolve())
        opt = {}
        for k, p in run.model.named_parameters():
            st = run.opt.state[p]
            opt.update({f"exp_avg/{k}": st["exp_avg"].cpu().numpy(), f"exp_avg_sq/{k}": st["exp_avg_sq"].cpu().numpy(),
                        f"step/{k}": np.array(float(st["step"]))})
        np.savez(session / "optimizer.npz", **opt)
    (session / "meta.json").write_text(json.dumps(meta, indent=1))
    write_history_md(session / "history.md", run, state)
    end = state["start"] + MAX_CONTINUE
    after = (f"Training then runs {ROUND_STEPS:,} more steps, after which there will be another round."
             if r < N_ROUNDS else
             f"This is the last round: training then runs without further edits until step {end:,}.")
    prompt = (ROOT / ("loop_prompt_v2.md" if V2 else "loop_prompt.md")).read_text(encoding="utf-8")
    for k, v in dict(SESSION=session.resolve(), ROOT=ROOT.resolve(), BUDGET=meta["budget"], TOTAL=total,
                     STEP=run.step, TRAIN_ACC=f"{m['train_acc']:.1%}", TEST_ACC=f"{m['test_acc']:.1%}",
                     ROUND=r, N_ROUNDS=N_ROUNDS, AFTER=after, N_TRAIN=len(run.train_pairs), END=f"{end:,}",
                     ROUND_STEPS=f"{ROUND_STEPS:,}",
                     **{k.upper(): v for k, v in CFG.__dict__.items()}).items():
        prompt = prompt.replace("{" + k + "}", str(v))
    (session / "PROMPT.md").write_text(prompt, encoding="utf-8")
    print(f"seed {seed}: round {r} session ready at step {run.step} "
          f"(train {m['train_acc']:.1%}, held-out {m['test_acc']:.1%}, firing {int(firing(run).sum())}/{CFG.hidden}): "
          f"{session}")


def record_round(run, r, before_w, sub_w, dead_before):
    """Per-round bookkeeping: budget used, and dead neurons revived by the edit itself."""
    alive_after = firing(run)
    return dict(round=r, step=run.step, changed=int(sum((sub_w[k] != before_w[k]).sum() for k in before_w)),
                dead_before=int(dead_before.sum()),
                revived_by_edit=int((dead_before & alive_after).sum()),
                firing_after_edit=int(alive_after.sum()))


def cmd_start(args):
    seed = args.seed
    out = loop_dir(seed)
    out.mkdir(parents=True, exist_ok=True)
    run = Run.load(seed_dir(seed) / "edit_point.pt")
    start = run.step
    if V2:
        state = dict(seed=seed, start=start, rounds=[])
        (out / "round1").mkdir(exist_ok=True)
        run.save(out / "round1" / "checkpoint.pt")
        save_state(seed, state)
        build_session(seed, 1, run, state)
        return
    dead = ~firing(run)
    before = get_weights(run.model)
    sub = dict(np.load(round_session(seed, 1) / "submission.npz"))
    apply_edit(run, sub, 1)
    state = dict(seed=seed, start=start, rounds=[record_round(run, 1, before, sub, dead)])
    run.train(ROUND_STEPS)
    state["rounds"][0]["firing_at_next_pause"] = int(firing(run).sum())
    (out / "round2").mkdir(exist_ok=True)
    run.save(out / "round2" / "checkpoint.pt")
    save_state(seed, state)
    build_session(seed, 2, run, state)


def cmd_next(args):
    seed = args.seed
    state = load_state(seed)
    r = len(state["rounds"]) + 1
    session = round_session(seed, r)
    if not (session / "submission.npz").exists():
        raise SystemExit(f"round {r} has no submission yet: {session}")
    run = Run.load(loop_dir(seed) / f"round{r}" / "checkpoint.pt")
    dead = ~firing(run)
    before = dict(np.load(session / "weights.npz"))
    sub = dict(np.load(session / "submission.npz"))
    apply_edit(run, sub, r)
    state["rounds"].append(record_round(run, r, before, sub, dead))
    if (session / "submission_optreset.npz").exists():
        state["rounds"][-1]["optimizer_resets"] = reset_moments(run, dict(np.load(session / "submission_optreset.npz")))
    if r < N_ROUNDS:
        run.train(ROUND_STEPS)
        state["rounds"][-1]["firing_at_next_pause"] = int(firing(run).sum())
        (loop_dir(seed) / f"round{r + 1}").mkdir(exist_ok=True)
        run.save(loop_dir(seed) / f"round{r + 1}" / "checkpoint.pt")
        save_state(seed, state)
        build_session(seed, r + 1, run, state)
        return
    run.train(state["start"] + MAX_CONTINUE - run.step)
    state["rounds"][-1]["firing_at_end"] = int(firing(run).sum())
    save_state(seed, state)
    res = dict(seed=seed, arm="opus_loop", rounds=state["rounds"], **summarize(run.history, state["start"]),
               notes=[json.loads((round_session(seed, i) / "submission.json").read_text())["note"]
                      for i in range(1, N_ROUNDS + 1)], history=run.history)
    (loop_dir(seed) / "results.json").write_text(json.dumps(res, indent=1))
    print(f"seed {seed} opus loop: {res['steps_to']}, best held-out {res['best_test']:.1%}")


def cmd_revival(args):
    """Opus round 1 (= the round 2 checkpoint), then re-init every never-firing neuron at +4k and train on.

    Re-init: fc1 row and bias with nn.Linear's default init (U(-1/sqrt(fan_in), 1/sqrt(fan_in))),
    fc2 column set to zero so the function is unchanged at the moment of revival. The AdamW moments
    of all re-initialized entries are reset to zero.
    """
    seed = args.seed
    state = load_state(seed)
    run = Run.load(loop_dir(seed) / "round2" / "checkpoint.pt")
    dead = np.where(~firing(run))[0]
    g = torch.Generator().manual_seed(3000 + seed)
    fc1, fc2 = run.model.fc1, run.model.fc2
    bound = 1 / math.sqrt(fc1.in_features)
    idx = torch.as_tensor(dead, device=fc1.weight.device)
    with torch.no_grad():
        fc1.weight[idx] = ((torch.rand(len(dead), fc1.in_features, generator=g) * 2 - 1) * bound).to(fc1.weight)
        fc1.bias[idx] = ((torch.rand(len(dead), generator=g) * 2 - 1) * bound).to(fc1.bias)
        fc2.weight[:, idx] = 0
        for p, sel in ((fc1.weight, (idx,)), (fc1.bias, (idx,)), (fc2.weight, (slice(None), idx))):
            for key in ("exp_avg", "exp_avg_sq"):
                run.opt.state[p][key][sel] = 0
    m = run.evaluate()
    m["edit"] = "revival"
    run.history.append(m)
    revived_firing = int(firing(run)[dead].sum())
    run.train(state["start"] + MAX_CONTINUE - run.step)
    res = dict(seed=seed, arm="revival", revival_step=m["step"] - state["start"],
               reinit=dead.tolist(), reinit_firing_right_after=revived_firing,
               fc2_init="zeros", **summarize(run.history, state["start"]), history=run.history)
    (loop_dir(seed) / "revival.json").write_text(json.dumps(res, indent=1))
    print(f"seed {seed} revival: re-initialized {len(dead)} neurons ({revived_firing} fire right after) | "
          f"{res['steps_to']}, best held-out {res['best_test']:.1%}")


def cmd_random(args):
    """At each round, a random edit with the same per-tensor entry count and delta L2 norm as Opus's edit."""
    seed = args.seed
    state = load_state(seed)
    if len(state["rounds"]) < N_ROUNDS:
        raise SystemExit("run the Opus loop to the end first")
    out = {}
    for d in range(N_RANDOM_DRAWS):
        rng = np.random.default_rng(4000 + 10 * seed + d)
        run = Run.load(seed_dir(seed) / "edit_point.pt")
        rounds = []
        for r in range(1, N_ROUNDS + 1):
            before = dict(np.load(round_session(seed, r) / "weights.npz"))
            sub = dict(np.load(round_session(seed, r) / "submission.npz"))
            cur = get_weights(run.model)
            target = {k: cur[k] + (sub[k] - before[k]) for k in cur}
            dead = ~firing(run)
            new = random_edit(cur, target, rng)
            apply_edit(run, new, r)
            rounds.append(record_round(run, r, cur, new, dead))
            f = round_session(seed, r) / "submission_optreset.npz"
            if f.exists():  # where Opus reset moments in a tensor, reset them at the random edit's changed entries
                opus_reset = dict(np.load(f))
                rounds[-1]["optimizer_resets"] = reset_moments(
                    run, {k: (new[k] != cur[k]) & bool(opus_reset[k].any()) for k in cur})
            run.train(ROUND_STEPS if r < N_ROUNDS else state["start"] + MAX_CONTINUE - run.step)
        res = dict(rounds=rounds, **summarize(run.history, state["start"]), history=run.history)
        out[f"draw{d}"] = res
        print(f"seed {seed} random loop draw {d}: {res['steps_to']}, best held-out {res['best_test']:.1%}")
    (loop_dir(seed) / "random_loop.json").write_text(json.dumps(dict(seed=seed, arm="random_loop", draws=out), indent=1))


def fmt(v):
    return "never" if v is None else f"{v:,}"


def cmd_report(args):
    lines = ["| Seed | Arm | Steps to 50% | Steps to 95% | Best held-out (step) |", "|---|---|---|---|---|"]
    for sd in sorted(RUNS.glob("seed*")):
        seed = int(sd.name[4:])
        pilot = json.loads((sd / "results.json").read_text())
        start = pilot["arms"]["unedited"]["history"][0]["step"]
        arms = []
        for name in ("unedited", "opus"):
            a = pilot["arms"][name]
            arms.append((name if name == "unedited" else "opus single edit", summarize(a["history"], start)))
        for loop, tag in (("loop", "loop"), ("loop2", "v2 loop")):
            ld = sd / loop
            for f, name in (("results.json", f"**opus {tag}**"), ("revival.json", "revival at +4k")):
                if (ld / f).exists():
                    arms.append((name, json.loads((ld / f).read_text())))
            if (ld / "random_loop.json").exists():
                for d, res in json.loads((ld / "random_loop.json").read_text())["draws"].items():
                    arms.append((f"random {tag} {d[-1]}", res))
        for name, s in arms:
            lines.append(f"| {seed} | {name} | {fmt(s['steps_to']['test50'])} | {fmt(s['steps_to']['test95'])} | "
                         f"{s['best_test']:.1%} ({s['best_test_step']:,}) |")
    print("\n".join(lines))
    print("\nper-round (opus loop): seed | round | changed | dead before | revived by edit | firing after edit")
    for f in sorted(RUNS.glob("seed*/loop*/results.json")):
        res = json.loads(f.read_text())
        for r in res["rounds"]:
            print(f"{f.parent.name} {res['seed']} | {r['round']} | {r['changed']} | {r['dead_before']} | "
                  f"{r['revived_by_edit']} | {r['firing_after_edit']} | resets {r.get('optimizer_resets', 0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "next", "revival", "random"):
        sub.add_parser(name).add_argument("--seed", type=int, required=True)
    sub.add_parser("report")
    args = ap.parse_args()
    V2 = args.v2
    dict(start=cmd_start, next=cmd_next, revival=cmd_revival, random=cmd_random, report=cmd_report)[args.cmd](args)
