# Next experiment: repeated Opus edits during training

This brief is written to be picked up by a fresh session. Read `README.md` (setup) and `RESULTS.md` (pilot 1) first.

## Why
In pilot 1, Opus 5.5 made one edit at the memorization plateau. Its runs reached 50% held-out 1.8–3.4× sooner than every control, in all 3 seeds, but then levelled off at 78–84% and never reached 95%.

The cause is known. Each agent switched off 54–62 "memorizing" neurons with a large negative bias, and none of those neurons ever fired again. A ReLU that never fires gets no gradient, and weight decay shrinks its bias and weights at the same rate, so the network is permanently left with 66–74 neurons. Random neuron removal levels off too, which confirms that the ceiling comes from lost capacity.

A single edit can't see its own consequences. This experiment tests the original idea instead: Opus edits **every X steps**, sees how training responded, and can correct itself, for example by putting dead neurons back to use.

The design rule is unchanged: Opus never uses an optimizer. Training is plain AdamW, and Opus only reads weights and writes values through `edit_tools.py`.

## Design

**Schedule:** edits at +0, +4k, +8k and +12k steps after the edit point (4 rounds), then plain training up to 60k steps after the first edit.
- **Round 1:** reuse each seed's pilot submission (`runs/seedN/agent/submission.npz`). Runs are fully reproducible (checked: same seed gives identical histories), so this is exactly the pilot's state.
- **Why 4k:** that's about where the pilot's Opus runs pass 50% held-out and begin to flatten.
- **New agent sessions:** rounds 2–4 × 3 seeds = 9.

**Every round is a fresh Opus 5.5 subagent,** not a long-running one. Its session folder holds the current weights and `history.md`. That file lists every earlier round's edit summary, its `submission.json` note, and the training/held-out accuracy curve since each edit (every 500 steps). Fresh agents are stateless, so each round can be replayed through the API from its files.

**Prompt:** start from `agent_prompt.md` and make minimal changes:
- Say which round this is, that training continues after the edit and that there will be further rounds, and to read `history.md`.
- Keep it neutral. **Do not** mention dead neurons, ReLU or revival. The agent can see "active frac 0.000" in the `stats` table, and whether it notices is part of the test.
- Keep rules 1–5 as they are.

**Budget:** 2% (749 entries) per round. Measure each round's budget against the checkpoint at the start of that round, not the original one. Each round's `weights.npz` is the current state.

**Optimizer state:** keep the AdamW moments across edits, as in the pilot, for every arm.

## Arms (seeds 0, 1, 2)

| Arm | Status | What it tells us |
|---|---|---|
| Unedited | already in `runs/seedN/results.json` | Baseline |
| Opus, single edit | already in `runs/seedN/results.json` | Pilot: fast early, stuck at about 80% |
| **Opus loop** (rounds at +0/4k/8k/12k) | new | The thing being tested |
| **Opus round 1, then scripted revival at +4k** | new, no agent needed | Whether restoring capacity is all later rounds need |
| **Random-edit loop** | new | Whether any repeated perturbation of the same size does this |

**Scripted revival:** at +4k, re-initialize every neuron that never fires on the training set:
- its fc1 row and bias with PyTorch's default `nn.Linear` init, and its fc2 column with zeros (or the default init; pick one and record it)
- reset the AdamW moments for those entries as well

Then train plainly.

**Random-edit loop:** at each round, apply `experiment.random_edit` matched to *that round's* Opus edit for the same seed. That means the same number of changed entries and the same total change (L2 norm) per tensor. Use 2 random draws per seed if time allows. It depends on the Opus loop's edits, so run it after the Opus loop.

## Pass bar (set before running)
- **Pass:** the Opus loop reaches 95% held-out at least 30% sooner than unedited in at least 2 of 3 seeds, and the random-edit loop doesn't.
- **Stronger result:** the Opus loop also beats scripted revival (reaches 95% sooner, or a higher best held-out score within 60k).
- Also report steps to 50%, best held-out score within 60k, budget used per round, and how many dead neurons were revived per round.

## Implementation: `loop.py`
Reuse `grok.py` (`Run`, `set_weights`, `get_weights`), `experiment.py` (`random_edit`, `continue_arm` logic, prompt rendering) and `edit_tools.py` as they are.

1. **`loop.py start --seed N`:** load `runs/seedN/edit_point.pt` and apply the pilot submission (round 1). Train 4,000 steps, recording history, then save `runs/seedN/loop/round2/checkpoint.pt` with the full optimizer state. Build the round 2 session in `runs/seedN/loop/round2/agent/`:
   - `weights.npz` and `working.npz`
   - `train_pairs.npy`
   - `meta.json`, with `step` and `state` updated and the history including the held-out curve
   - `history.md`
   - a rendered `PROMPT.md`, with `{ROUND}` and the round's paths filled in
2. **Agent turn:** spawn an Opus 5.5 subagent (`Agent` tool, `model: "opus"`, `subagent_type: "general-purpose"`) with that round's `PROMPT.md` pasted word for word. The 3 seeds can run in parallel.
3. **`loop.py next --seed N`:** load the round's checkpoint, apply that round's `submission.npz`, train 4,000 steps, and build the next round. After round 4, train to the 60k cap and write `runs/seedN/loop/results.json`.
4. **`loop.py revival --seed N`** and **`loop.py random --seed N`:** the two control arms. Each writes its own results JSON.
5. **`loop.py report`:** table across all arms and seeds, plus an update to `RESULTS.md`.

`edit_tools.py` reads `weights.npz` as the base for its budget, so per-round budgets work unchanged if each round's `weights.npz` is that round's starting state.

## Audit after each round
Same as pilot 1:
- Check `tool_log.jsonl` and `scratch/*.py` for any of the following: gradient descent, `optim`, `backward`, least-squares fitting, forward passes on held-out pairs, or automated scoring of many candidates. Look for `split(`, `test_pairs`, `lstsq`, `.train(`.
- Note gray areas in `RESULTS.md`. Pilot 1 had one: seed 2 scored a grid of about 25 kill-threshold variants.

## Cost and time
- **Agents:** 9 sessions of about 5–12 minutes and 100–150k tokens each, so roughly 1–1.5M tokens.
- **GPU:** each 4k-step segment takes about 5 seconds, and a full 60k continuation about 70 seconds, on an RTX 4070.

## If it passes
1. **Contamination check:** rerun with the input tokens secretly relabeled by a random shuffle, so the Fourier structure is still there but scrambled.
2. A small CNN with a planted problem.
3. A real fine-tuning run.

## Gotchas from pilot 1
- `kill_controls.py` treats a neuron as switched off if its bias changed **and** is now below -1. Seed 0 also rewrote biases of its partner neurons, which shouldn't count.
- The project folder's git root is the user's home directory (`<home>`). Don't commit from here without asking.
- Use `.venv\Scripts\python.exe` (torch 2.14 with CUDA, RTX 4070). `grok.py` sets `CUBLAS_WORKSPACE_CONFIG` and deterministic algorithms, so keep it imported first.
