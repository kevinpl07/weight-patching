# Results: pilot 1, a single Opus 5.5 edit at the memorization plateau (2026-11-03)

Three seeds. Each edit is made about 1,000 steps after training accuracy reaches 100%, while held-out accuracy is at 0.4–0.6%. Step counts are steps after the edit. Every arm ran at most 60k steps.

| Seed | Arm | Train / held-out right after | Steps to 50% held-out | Steps to 95% | Best held-out |
|---|---|---|---|---|---|
| 0 | unedited | 100% / 0.4% | 11,100 | 30,900 | 95.1% |
| 0 | **Opus** (59 neurons switched off, 5 partner neurons added, 681 entries) | 23.4% / 3.0% | **3,300** | never | 84.3% |
| 0 | random edit ×3 | 3–4% / ~1% | 9,500–12,000 | 53,200 / never / never | 87–95% |
| 0 | random neuron removal ×3 | 20–22% / ~0.8% | 6,500–8,300 | never | 86–91% |
| 1 | unedited | 100% / 0.4% | 7,000 | 22,900 | 95.1% |
| 1 | **Opus** (62 neurons switched off, 62 entries) | 27.5% / 3.2% | **3,400** | never | 78.5% |
| 1 | random edit ×3 | 1–2% / ~1% | 9,700–13,200 | never | 87–93% |
| 1 | random neuron removal ×3 | 20–24% / ~0.8% | 6,400–9,200 | never | 83–86% |
| 2 | unedited | 100% / 0.6% | 8,100 | 28,900 | 95.0% |
| 2 | **Opus** (54 neurons switched off, 54 entries) | 35.4% / 2.8% | **4,500** | never | 82.3% |
| 2 | random edit ×3 | 4–5% / ~0.8% | 10,400–13,300 | 50,500 / never / never | 84–95% |
| 2 | random neuron removal ×3 | 24–28% / ~0.8% | 6,700–8,200 | never | 86–90% |

- **Random edit:** changes the same number of entries per tensor by the same total amount (L2 norm of the change), at random positions and in random directions.
- **Random neuron removal:** switches off the same number of randomly chosen neurons with the same bias values. It was added after the first results, to separate "which neurons were removed" from "neurons were removed at all".

## Findings
- **The pre-set pass bar failed.** The bar was reaching 95% at least 30% sooner in 2 of 3 seeds. No Opus arm reached 95% at all.
- **Early on, the edit clearly helps.** Opus reaches 50% held-out 1.8–3.4× sooner than the unedited run, in all 3 seeds. It also beats every random edit and every random neuron removal. One thousand steps after the edit, Opus is at 14–20% held-out, against 2–3% for the unedited run.
- **What caps it:**
  - All three agents independently chose the same strategy. They switched off the neurons they judged to be memorizing by setting a large negative bias.
  - Those neurons never fire again: 0 of 54–62 did, in 60k steps. A ReLU that never fires gets no gradient, and weight decay shrinks its bias and weights at the same rate.
  - The network is left with 66–74 working neurons and levels off at 78–84%.
  - Random neuron removal levels off too (83–91%), so the ceiling comes from losing capacity, not from which neurons Opus picked.
  - The agents named this risk themselves, but they had no way to test the edit by continuing training.
- **The agents' predictions were off.** All three expected an immediate jump to 10–25% held-out; the actual figure was about 3%. They also expected 95% within about 1,500 steps, which never happened.

## Rule audit
- All forward passes used training pairs only. There was no gradient descent or fitting.
- **Gray area:** seed 2 scored a small hand-built grid of about 25 kill-threshold variants on training accuracy, then chose one by reasoning.
- Seed 0 computed a weights-only alignment score over the (a, b) grid using the a+b rule. That doesn't use held-out data.

## Scripts
- `experiment.py forks`: the unedited, Opus and random-edit arms.
- `kill_controls.py`: the dead-neuron check and the random neuron removal arms.
- Results are in `runs/seedN/results.json` and `runs/seedN/kill_controls.json`. Agent sessions, scripts and tool logs are in `runs/seedN/agent/`.

# Results: pilot 2, repeated Opus 5.5 edits during training (2026-11-03)

Design is in `NEXT_EXPERIMENT.md`. Opus edits at +0 (the pilot 1 submission), +4k, +8k and +12k steps after the edit point, then plain training to +60k. Each round is a fresh Opus 5.5 session with the same rules and a 749-entry budget, measured from that round's checkpoint. It sees `history.md`: earlier edits, their notes, and the training/held-out curve since each edit. The prompt does not mention dead neurons. Step counts are steps after the first edit. Each arm continues from the same checkpoint with the same AdamW state.

| Seed | Arm | Steps to 50% held-out | Steps to 95% | Best held-out (step) |
|---|---|---|---|---|
| 0 | unedited | 11,100 | 30,900 | 95.1% (stopped at 95%) |
| 0 | Opus, single edit | 3,300 | never | 84.3% |
| 0 | **Opus loop** | 3,300 | never | 86.2% (8,100); 80.2% at +60k |
| 0 | revival at +4k | 3,300 | 40,800 | 96.6% |
| 0 | random-edit loop ×2 | 10,500 / 10,500 | 31,000 / 52,900 | 99.0% / 96.7% |
| 1 | unedited | 7,000 | 22,900 | 95.1% (stopped at 95%) |
| 1 | Opus, single edit | 3,400 | never | 78.5% |
| 1 | **Opus loop** | 3,400 | never | 76.7% (12,300); 73.4% at +60k |
| 1 | revival at +4k | 3,400 | 24,100 | 98.7% |
| 1 | random-edit loop ×2 | 10,800 / 9,200 | never / 42,000 | 93.5% / 96.7% |
| 2 | unedited | 8,100 | 28,900 | 95.0% (stopped at 95%) |
| 2 | Opus, single edit | 4,500 | never | 82.3% |
| 2 | **Opus loop** | 4,000 | never | 90.9% (12,400); 87.4% at +60k |
| 2 | revival at +4k | 7,200 | 24,600 | 99.0% |
| 2 | random-edit loop ×2 | 8,300 / 10,700 | never / 52,000 | 86.3% / 95.3% |

- **Revival at +4k:** the Opus round-1 run up to +4k. Then every neuron that never fires on the training set gets its fc1 row and bias re-initialized with PyTorch's default `nn.Linear` init. Its fc2 column is set to **zeros**, and the AdamW moments for those entries are reset. 59, 62 and 54 neurons were re-initialized; all of them fire right after.
- **Random-edit loop:** at each round, a random edit with the same per-tensor entry count and delta L2 norm as that round's Opus edit, applied to the random arm's own current weights. Two draws per seed.

## Per round (Opus loop)

| Seed | Round | Entries changed | Dead before | Revived by the edit | Firing after the edit | Held-out right after the edit → 4k later |
|---|---|---|---|---|---|---|
| 0 | 1 | 681 | 0 | 0 | 69 | 3% → 55% |
| 0 | 2 | 749 | 59 | 0 | 54 | 69% → 75% |
| 0 | 3 | 749 | 74 | 0 | 54 | 84% → 77% |
| 0 | 4 | 749 | 74 | 2 | 56 | 82% → 81%, then flat at 80% |
| 1 | 1 | 62 | 0 | 0 | 66 | 3% → 55% |
| 1 | 2 | 688 | 62 | 0 | 57 | 59% → 72% |
| 1 | 3 | 686 | 71 | 7 | 64 | 63% → 70% |
| 1 | 4 | 592 | 64 | 3 | 60 | 70% → 73%, then flat at 73% |
| 2 | 1 | 54 | 0 | 0 | 74 | 3% → 47% |
| 2 | 2 | 748 | 54 | 0 | 58 | 58% → 77% |
| 2 | 3 | 128 | 70 | 70 | 128 | 78% → dropped to 47%, then 59% |
| 2 | 4 | 744 | 0 | 0 | 84 | 83% → 88%, then flat at 87% |

## Findings
- **The pre-set pass bar failed.** No Opus loop run reached 95% held-out within 60k steps (0 of 3). Two of the six random-loop draws didn't reach 95% either. The four that did reached it in 31k–53k steps, about as fast as or slower than unedited.
- **The stronger comparison also fails.** Scripted revival beats the Opus loop in every seed: it reaches 95% in all 3 (24.1k–40.8k steps) with a best of 96.6–99.0%. The Opus loop peaks at 77–91%.
- **Scripted revival is not faster than unedited**: 40.8k vs 30.9k, 24.1k vs 22.9k, and 24.6k vs 28.9k. Restoring capacity removes pilot 1's ceiling, but it doesn't make the run reach 95% sooner.
- **The Opus loop freezes.** From about +16k to +60k, held-out moves by less than 0.5 points in every seed (seed 0: 80.4% → 80.2%). Plain training doesn't keep generalizing after the last edit. The ceiling is not only a neuron count: seed 2 ends with 84 working neurons and still sticks at 87%.
- **What the agents did:**
  - In rounds 2 and 3 they mostly switched off more "memorizing" neurons and cleaned the Fourier neurons' leftover weights. Each edit gave an immediate jump of 4–14 points held-out. Over the next 4k steps, training mostly took that back: seed 0 round 3 went from 84% to 77%.
  - Round-2 agents switched off 9–16 more neurons each, so the network went down to 54–58 working neurons.
- **Dead neurons were noticed late, and not always believed fixable:**
  - All 8 sessions that started with dead neurons saw them in `stats` (seed 2 round 4 started with none).
  - Seed 0's agents in rounds 3 and 4 said gradient descent "cannot" bring them back. The round-4 agent thought it could revive only 2 within budget, even though resetting a bias costs 1 entry per neuron.
  - Only seed 2 round 3 revived all of them, at 1 bias entry each. The 70 revived neurons first learned to memorize (training loss 1e-4) and held-out fell from 78% to 47%. Round 4 then switched 44 of them off again and built constant "offset neurons", which led to the best Opus result (91%).
- **A new mechanism the agents found:**
  - In rounds 3–4, seeds 1 and 2 each concluded on their own that the Fourier circuit needs a per-class logit offset of about ±3–10.
  - Weight decay keeps fc2.bias within about ±1, so it can't hold that offset. Memorizing neurons were carrying it.
  - Both built always-on neurons to carry the offset. It helped a little (seed 2: +5 points) but didn't get past the plateau.
- **Predictions were again too optimistic:**
  - Every agent from round 2 on expected a lasting rise, often to "85–95%" or "past 95%".
  - Seed 2 round 4 gave 75% odds of a large jump and 55% odds of settling above 95%. The jump came (83%), but it settled at 87%.
  - The agents were right about the immediate effect of their edits. They were consistently wrong about what training does afterwards.

## Rule audit (rounds 2–4, 9 sessions)
- No gradient descent, no optimizer, no training, and no forward passes on held-out pairs. No script references `test_pairs` or `split(`.
- The `meshgrid` and `linspace` grids in seed 1 round 2 and seed 2 round 2 are analytic models over continuous angles. They involve no data.
- **Gray areas:**
  - **Small hand-built sweeps scored on training accuracy, then chosen by reasoning:**
    - seed 0 round 2: `denoise.py`, 9 variants
    - seed 2 round 2: several threshold sweeps of 4–9 values each
    - seed 1 round 3: a sweep of 5 λ values
    - seed 2 round 4: `cands.py`, a 4×7 grid of threshold × bias shift, the largest
  - Like pilot 1's seed 2, these sit at the edge of rule 1's "comparing a few hand-designed candidates".
  - **`lstsq` in seed 0 round 3 (`resid.py`):** used to project weight vectors onto a Fourier basis, which is the same as an FFT filter. It uses no data, so it is not fitting.

## Cost
9 agent sessions of about 6–10 minutes and 117–141k tokens each, about 1.15M tokens in total. GPU time is under 10 minutes in total.

## Scripts
- `loop.py`: `start`, `next`, `revival`, `random` and `report`. `loop_prompt.md` is the round prompt, a minimal change from `agent_prompt.md`.
- Per seed in `runs/seedN/loop/`: `results.json` (Opus loop), `revival.json`, `random_loop.json` and `state.json`. `roundK/agent/` holds each session, with `history.md`, `PROMPT.md`, `tool_log.jsonl`, `scratch/` and the submission. `roundK/checkpoint.pt` is the full run state at each pause.

## Capacity check: narrower networks trained from scratch
`capacity_check.py`: same task and settings, hidden width set to the working-neuron counts the Opus loop ended with, trained from step 0 for up to 62,300 steps. Results are in `runs/capacity_h*.json`.

| Hidden width | Reaches 95% held-out | Best held-out (seeds 0 / 1 / 2) | Opus loop ending at this many working neurons |
|---|---|---|---|
| 54 | never | 76.7% / 76.3% / 78.1% | seed 0: 56 neurons, 80.2% |
| 60 | never | 89.7% / 85.1% / 83.1% | seed 1: 60 neurons, 73.4% |
| 84 | never | 92.0% / 80.3% / 82.9% | seed 2: 84 neurons, 87.4% |
| 128 | 25k–33k steps | 95% (the unedited baseline) | |

- **Capacity explains most of the ceiling.** None of the 9 narrower networks reaches 95% within the budget, even trained cleanly from scratch. At 54–84 neurons this task tops out around 76–92%, and the Opus loop's endings fall in the same range.
- **Exception:** seed 1's loop ended about 10–16 points below a clean 60-neuron network, so its edits did harm beyond lost capacity.
- **Width 128 is close to the minimum for grokking here.** Any strategy that removes neurons, including Opus's "switch off the memorizers" in every round 1, costs generalization this network can't spare. In a much wider network the same strategy might be harmless.

# Results: pilot 3, v2 prompt and tools (2026-11-03)

Same loop as pilot 2 (4 rounds, 4k steps apart, then train to +60k), but round 1 is a fresh edit at the edit point. Changes:
- **Prompt (`loop_prompt_v2.md`):**
  - the edit is "a starting point for the optimizer" and is judged by where training goes after it
  - "anything you remove, it may be unable to bring back"
  - "find what is slowing the optimizer down"
  - each submit includes predictions of held-out at +4k and at the end; `history.md` shows them next to what actually happened
- **Tools (`edit_tools.py`):**
  - `optim`: read-only view of the gradient, the AdamW moments and the next update
  - `reset_optimizer`: an edit op that zeroes moments, only on changed entries, at no budget cost
  - `trial`: at most 3 throwaway runs of up to 1,000 steps each, showing training and held-out curves; checked to match the real run exactly
- **Rule 1 now says:** read gradients freely, but don't write values made by stepping along them.
- **Random-edit control:** matched to each v2 round. Where Opus reset moments in a tensor, it resets them at its own changed entries.

| Seed | Arm | Steps to 50% | Steps to 95% | Best held-out | Mean held-out over the first 12k | Mean held-out over 60k |
|---|---|---|---|---|---|---|
| 0 | unedited | 11,100 | 30,900 | 95.1% | 20.9% | ≥76.7%* |
| 0 | Opus loop, pilot 2 | 3,300 | never | 86.2% | 62.2% | 76.7% |
| 0 | **Opus loop, v2** | 4,100 | never | 78.9% | 49.4% | 72.5% |
| 0 | random loop matched to v2 ×2 | 7,800 / 9,400 | 37,400 / 34,200 | 98.8% / 98.2% | 33.8% / 27.4% | 81.2% / 79.2% |
| 1 | unedited | 7,000 | 22,900 | 95.1% | 39.2% | ≥82.8%* |
| 1 | Opus loop, pilot 2 | 3,400 | never | 76.7% | 58.1% | 70.4% |
| 1 | **Opus loop, v2** | 4,100 | never | 85.9% | 54.9% | 75.7% |
| 1 | random loop matched to v2 ×2 | 6,800 / 7,600 | 36,900 / 30,000 | 97.3% / 97.1% | 41.2% / 37.1% | 83.1% / 82.9% |
| 2 | unedited | 8,100 | 28,900 | 95.0% | 34.1% | ≥80.8%* |
| 2 | Opus loop, pilot 2 | 4,100 | never | 90.9% | 52.6% | 80.4% |
| 2 | **Opus loop, v2** | 2,900 | never | 93.0% | 63.1% | **86.7%** |
| 2 | random loop matched to v2 ×2 | 10,300 / 9,500 | 50,900 / 39,100 | 96.2% / 97.7% | 22.6% / 27.1% | 74.7% / 77.6% |

\*Unedited stops at 95%, so I held it at 95% for the rest of the run; that makes it a lower bound.

## Per round (v2)

| Seed | Round | Entries changed | Optimizer resets | Dead neurons before | Revived by the edit | Firing after the edit | Predicted at +4k | Actual at +4k | Predicted at end |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 183 | 97 | 0 | 0 | 42 | 75% | 44.0% | 97% |
| 0 | 2 | 749 | 0 | 86 | 0 | 42 | 52% | 54.9% | 62% |
| 0 | 3 | 749 | 582 | 86 | 2 | 44 | 60% | 59.4% | 63% |
| 0 | 4 | 738 | 738 | 84 | 11 | 55 | 80% | 78.6% | 90% → actual 78.6% |
| 1 | 1 | 717 | 0 | 0 | 0 | 128 | 15% | 18.9% | 97% |
| 1 | 2 | 745 | 59 | 0 | 0 | 69 | 82% | 78.6% | 97% |
| 1 | 3 | 749 | 0 | 59 | 0 | 69 | 81% | 79.9% | 92% |
| 1 | 4 | 694 | 0 | 59 | 5 | 74 | 81% | 79.4% | 82% → actual 81.4% |
| 2 | 1 | 75 | 75 | 0 | 0 | 53 | 85% | 56.8% | 97% |
| 2 | 2 | 747 | 747 | 75 | 6 | 56 | 76% | 72.1% | 90% |
| 2 | 3 | 742 | 742 | 72 | 14 | 70 | 84% | 85.6% | 90% |
| 2 | 4 | 726 | 726 | 58 | 12 | 82 | 91% | 93.0% | 93% → actual 92.8% |

## Findings
- **The pass bar failed again.** No v2 run reached 95% (0 of 3). All 6 random-loop draws did, at 30k–51k steps, about as fast as or slower than unedited.
- **v2 is better than pilot 2 in 2 of 3 seeds:**
  - end held-out 81.4% vs 73.4% (seed 1), and 92.8% vs 87.4% (seed 2)
  - seed 2 v2 is the best Opus run so far, and the only one ahead of unedited on mean held-out over the 60k steps
  - seed 0 is worse (78.6% vs 80.2%), because of its round-1 cut
- **The prompt didn't stop the neuron cuts. The trials encouraged them:**
  - Seed 0 cut 86 neurons and seed 2 cut 75 in round 1. Seed 1 cut 59 in round 2.
  - A 1,000-step trial makes a cut look great (held-out 25% vs 2%, or 74% vs 25%), and its cost appears only after 10k+ steps.
  - Seed 2 round 1 chose its cut-off by comparing three trials on held-out accuracy.
- **What the new tools did help with:**
  - **Rejecting bad revivals.** Seeds 0 and 2 tried switching all the dead neurons back on, saw in a trial that they memorized (held-out fell to 12–30%), and undid it.
  - **Rebuilding capacity on purpose.** From round 2 on, seeds 0 and 2 rebuilt dead neurons as noise-cancelling partners: phase-shifted copies at existing frequencies, with optimizer resets.
  - **Steady gains.** Unlike pilot 2's jump-then-decay, these edits mostly gave gains that held, e.g. seed 2: 57% → 72% → 86% → 93%.
- **Prediction feedback made the agents accurate:**
  - Round 1, before any feedback: off by 4–31 points.
  - Rounds 2–4: all 9 within 4 points.
  - End-of-run predictions became realistic by round 4 for seeds 1 and 2 (82% vs 81.4%, 93% vs 92.8%). Seed 0 still overshot (90% vs 78.6%).
- **Capacity still sets the ceiling.**
  - Ending neuron counts were 55, 74 and 82, with held-out 78.6%, 81.4% and 92.8%. That matches networks of those widths trained from scratch (54: about 77%; 84: 80–92%).
  - The agents rebuilt neurons at about 2–14 per round. The budget of 749 entries limits that: a fully seeded neuron costs about 100–300 entries.

## Rule audit
- No gradient-stepped values, no fitting, and no forward passes on held-out pairs in any session's scripts. The grep hits are argument parsing or analytic grids over angles.
- **By design:** trials report held-out accuracy, and agents used it to choose between their up to 3 candidates per round, e.g. seed 2 round 1's cut-off. That is a small amount of selection on the test set; a cleaner rerun would show a separate validation split in trials.

## Cost
12 agent sessions of 92–136k tokens each (about 1.4M total) and 4–11 minutes each.


# Results: an agent's edit at the start of LLM RL training (2026-11-04)

Plan and pre-registered pass bar: [PLAN_2026-11-03_rl_head_start.md](PLAN_2026-11-03_rl_head_start.md), including three amendments made during the run. Setup: [rl/README.md](rl/README.md).

- **Model and training:** Qwen3.5-0.8B, non-thinking mode, full-weight GRPO on GSM8K: 8 problems × 8 samples per step, 512 tokens, lr 1e-6, no weight decay, 8-bit AdamW, 100 steps.
- **Reward:** 1 for a correct last `\boxed{}` answer, plus 0.1 for any boxed answer. With `--trunc-zero` (Amendment 2), an answer that hits the 512-token limit gets 0.
- **Arms:**
  - **A:** base model.
  - **B:** base plus the agent's edit from `runs/agent_dry/model_edit` (an anti-hedging steer: 4,089 weights in 4 `down_proj` columns).
  - **C:** base plus a random edit of the same size (`runs/agent_dry/model_random`).
  - **D:** base model sampled at T=0.8 during training.
- **Quick eval** at steps 0, 15, 30, 50 and 100:
  - GSM-Symbolic: 500 problems, scored greedy and sampled at T=1 (the primary metric).
  - IFEval: the first 200 prompts, scored greedy.

## Quick evals

GSM-Symbolic sampled at T=1 (primary metric):

| Run | Step 0 | 15 | 30 | 50 | 100 |
|---|---|---|---|---|---|
| A, seed 0 | 26.8% | 42.6% | 49.4% | 51.8% | 54.4% |
| A, seed 1 | 26.8% | 44.2% | 51.2% | 60.4% | 58.6% |
| B, seed 0 | **34.0%** | 42.2% | 44.4% | 53.6% | 55.6% |
| B, seed 1 | **34.0%** | 44.4% | 53.4% | 51.8% | 56.8% |
| C, seed 0 | 26.0% | 37.4% | 50.0% | 56.0% | 55.6% |
| D, seed 0 (invalid, see below) | 26.8% | 23.2% | 17.2% | 1.4% | 28.6% |

GSM-Symbolic greedy:

| Run | Step 0 | 15 | 30 | 50 | 100 |
|---|---|---|---|---|---|
| A, seed 0 | 47.2% | 49.4% | 53.8% | 54.6% | 57.6% |
| A, seed 1 | 47.2% | 54.4% | 57.8% | 60.2% | 65.2% |
| B, seed 0 | 43.4% | 52.6% | 50.4% | 58.2% | 55.0% |
| B, seed 1 | 43.4% | 48.6% | 57.4% | 61.2% | 60.4% |
| C, seed 0 | 45.4% | 42.6% | 54.0% | 58.2% | 60.2% |

IFEval prompt-level strict (200 prompts):

| Run | Step 0 | 15 | 30 | 50 | 100 |
|---|---|---|---|---|---|
| A, seed 0 | 50.0% | 52.0% | 54.0% | 55.5% | 55.5% |
| A, seed 1 | 50.0% | 53.0% | 47.5% | 53.0% | 49.0% |
| B, seed 0 | 55.0% | 54.0% | 49.5% | 49.0% | 54.0% |
| B, seed 1 | 54.5% | 50.0% | 48.5% | 48.0% | 49.5% |
| C, seed 0 | 51.0% | 52.0% | 56.5% | 57.5% | 52.0% |

Truncation on GSM-Symbolic sampled fell from 57% (A, C) and 43% (B) at step 0 to 1–13% for every valid run by step 15, and stayed there. No run with `--trunc-zero` showed the answer-repeat collapse.

## B − A, averaged over seeds 0 and 1

Each seed's B is compared with the A of the same seed, which has the same data order.

| Metric | Step 0 | 15 | 30 | 50 | 100 |
|---|---|---|---|---|---|
| Sampled (primary) | **+7.2** | −0.1 | −1.4 | −3.4 | −0.3 |
| Greedy | −3.8 | −1.3 | −1.9 | +2.3 | −3.7 |
| IFEval | +4.8 | −0.5 | −1.8 | −5.8 | −0.5 |

## Pass bar (Amendment 1 version)

| Condition | Result | Verdict |
|---|---|---|
| 1. Head start: B − A averaged over steps 15 and 30 is at least +5 | −0.8 (seed 0: −2.7, seed 1: +1.2) | **Fail** |
| 2. Ends better: B − A averaged over steps 50 and 100 is at least +3 | −1.8 (seed 0: +1.5, seed 1: −5.2) | **Fail** |
| 3. No damage: at step 100, B no more than 2 points below A on greedy and on IFEval | greedy −3.7, IFEval −0.5 | **Fail** (greedy) |
| 4. C does not meet condition 1 | C − A at step 15 and 30: −5.2, +0.6 | Met |
| 5. B meets condition 1 against D | D invalid | Not tested |

**"B helps training" needs 1, 3 and 4, so it fails.**

## Training batches

Training accuracy / truncation / mean length (tokens), over windows of steps. Both arms with the same seed see the same problems in the same order.

| Steps | A, seed 0 | B, seed 0 | A, seed 1 | B, seed 1 | C, seed 0 |
|---|---|---|---|---|---|
| 1–5 | 38% / 33% / 373 | 43% / 25% / 388 | 41% / 36% / 379 | 46% / 22% / 371 | 37% / 30% / 371 |
| 6–10 | 40% / 12% / 289 | 41% / 22% / 366 | 53% / 12% / 278 | 45% / 10% / 346 | 42% / 11% / 312 |
| 11–15 | 41% / 5% / 242 | 39% / 7% / 299 | 52% / 8% / 244 | 49% / 14% / 355 | 38% / 4% / 274 |
| 16–30 | 51% / 4% / 214 | 43% / 2% / 210 | 52% / 8% / 220 | 51% / 7% / 301 | 46% / 2% / 226 |
| 31–50 | 57% / 1% / 192 | 56% / 2% / 249 | 63% / 7% / 234 | 60% / 4% / 241 | 60% / 3% / 218 |
| 51–100 | 59% / 3% / 199 | 62% / 3% / 244 | 66% / 6% / 264 | 60% / 7% / 258 | 63% / 5% / 246 |

## Findings

- **The edit's head start is real but small for RL.**
  - At step 0, B leads A by 7.2 points on sampled accuracy, while the random edit C sits at −0.8. The independent scoring in `runs/eval/` had measured +13.2; the two use different sampling seeds and batching.
  - B also leads in training accuracy over steps 1–5 on both seeds (+5 points).
  - The edit's greedy cost (−3.8) and IFEval gain (+4.8) at step 0 are both near the earlier measurements.
- **Training erases it in about 10 steps.**
  - A gains 16–17 points by step 15, so B's 7-point lead is worth about 7 steps.
  - Averaged over seeds, B − A is −0.1 at step 15 and stays within ±3.5 points on sampled accuracy for the rest of the run.
- **After the start, B looks like a random perturbation.** At step 100, B seed 0 and C seed 0 both end +1.2 above A seed 0. Over steps 51–100, C's training accuracy (63%) is higher than B's (62%).
- **The only consistent B-vs-A difference: B shortens later.** Over steps 6–15, both B seeds wrote longer answers than their A (B: 366/299 and 346/355 tokens; A: 289/242 and 278/244). A likely reason: B starts with fewer truncated answers, so fewer zero-reward samples push it toward ending. It made no difference to accuracy and was gone by step 30 to 50.
- **The greedy cost may persist.** At step 100, B is below A on greedy in both seeds (−2.6, −4.8), about the same as the step-0 cost (−3.8). The two A seeds end 7.6 points apart on greedy, so this is weak evidence.
- **Run-to-run noise is large.** The two A seeds end 4.2 points apart on sampled accuracy, 7.6 on greedy and 6.5 on IFEval (the latter in the opposite direction), and were 8.6 points apart on sampled at step 50. Single-seed dips during the run (B0 at step 30, C0 at step 15, B0's IFEval at steps 30–50) all recovered. One seed per arm can't resolve differences of a few points here.
- **GRPO fixes the base model's rambling almost by itself.** Truncation goes from 57% to about 3% within 15 steps, and sampled accuracy from 27% to about 43%. The plan had predicted A would need 100–150 steps to catch B.

## What went wrong, and what was fixed

1. **Reward flaw (Amendment 2).**
   - **What happened:** the first arm A (300-step schedule, `runs/grpo/armA_s0`) collapsed after step 89. It wrote the answer, `</think>`, and the answer again, then stopped emitting the end token and repeated until the 512-token limit. Truncation went from 2% to 100% between steps 89 and 96, while training accuracy stayed at about 62%.
   - **Why:** the reward scored the last boxed answer and ignored truncation.
   - **Fix:** `--trunc-zero`. No later run collapsed.
2. **Arm D temperature bug (Amendment 3).**
   - **What happened:** `grpo.py` sampled at T=0.8 but took gradients on T=1 log-probs. The never-sampled token tail grew until D collapsed: 1.4% sampled accuracy at step 50, 3% training accuracy over steps 41–50, and junk tokens in its outputs.
   - **Fix:** `token_logprobs` now divides the logits by T (no change at T=1). The fixed D was not rerun.
3. **Schedule change (Amendment 1).** The original plan was 300 steps with evals every 75 steps. After arm A's step 75, it was clear the whole A-vs-B comparison happens in the first 30 steps, so all arms were rerun at 100 steps with evals at 0, 15, 30, 50 and 100.

## Not done

- **The fixed arm D**, so condition 5 (does the edit beat sampling at T=0.8) is untested.
- **The hedging-direction analysis** (`rl/ckpt_analysis.py`, written and tested on `model_edit`/`model_random` only), which would show whether RL moves along the agent's direction by itself. Every checkpoint it needs is saved.
- **Full scoring** (`rl/eval.py`) of the step-100 checkpoints.

## Files

- **Runs:** `runs/grpo/arm{A,B,C,D}_s{0,1}_100/`. Each has `config.json`, `log.jsonl`, `evals.jsonl`, `evals/` (every eval answer), `rollouts/` and `ckpt/step_000{15,30,50}`, `ckpt/step_00100` (about 12 GB per run).
- **Collapsed first arm A:** `runs/grpo/armA_s0/` (step-75 checkpoint and all rollouts).
- **Driver log:** `runs/grpo/arms.log`; the first, stopped driver is in `runs/grpo/arms_s0.log`.
- **Code:** `rl/grpo.py` (`--init`, `--eval-at`, `--trunc-zero`, temperature-scaled log-probs), `rl/run_arms.ps1`, `rl/evals_table.py`, `rl/ckpt_analysis.py`.

# Results: agent edits during small-LM pretraining (2026-11-04)

Pretraining from scratch on one RTX 4070, looking for a loss plateau where an edit might beat the optimizer. All runs use seed 0. The summary is in `FINDINGS.md`; the scripts are described in `pretrain/README.md`.

## Where is there a plateau? (baseline grid)

- **Training settings for every run:** 15,000 steps, 64 × 512 tokens per step, AdamW lr 1e-3 (100-step warmup, cosine decay to 1e-4), weight decay 0.1. The TinyStories runs used a 256-token context and 64 × 256 tokens per step.
- **Tokenizers:** byte-level BPE with 4,096 tokens (TinyStories) or 8,192 (web text and code).
- **Probe columns:** "onset" is the first probe where the smoothed copy benefit reaches 0.25 nats (copy benefit = loss on the first copy of a real passage minus loss on an exact repeat). "t90" is the step where it reaches 90% of its final value.

| Data | Model | Params | Onset | t90 | Final copy benefit | Val @1k | Val @3k | Val @15k |
|---|---|---|---|---|---|---|---|---|
| TinyStories | 2L attention-only d256 | 2.7M | never | – | 0.12 | 2.484 | 2.159 | 1.866 |
| TinyStories | 4L GPT d256 | 5.3M | never | – | 0.09 | 2.319 | 1.877 | 1.499 |
| Web (FineWeb-Edu) | 2L attention-only d256 | 4.9M | **5,350** | 11,200 | 0.96 | 4.785 | 4.458 | 3.926 |
| Web | 2L attention-only d512 | 10.8M | 1,600 | 4,200 | 1.63 | 4.932 | 4.073 | 3.671 |
| Web | 4L GPT d256 | 7.5M | 1,675 | 4,525 | 1.39 | 4.702 | 3.967 | 3.492 |
| Web | 6L GPT d384 | 17.1M | 1,000 | 4,525 | 1.73 | 4.432 | 3.732 | 3.249 |
| Code (codeparrot-clean) | 2L attention-only d256 | 4.9M | 1,975 | 3,675 | 1.13 | 3.150 | 2.485 | 2.088 |
| Code | **2L attention-only d512** | 10.8M | **1,875** | **2,450** | 1.19 | 3.114 | 2.351 | 1.911 |
| Code | 4L GPT d256 | 7.5M | 950 | 2,125 | 1.24 | 2.916 | 2.220 | 1.773 |
| Code | 6L GPT d384 | 17.1M | 700 | 1,100 | 1.16 | 2.559 | 2.037 | 1.585 |

- **TinyStories:** induction never forms (best induction score 0.06) and the loss curve is smooth. The data rarely rewards exact copying.
- **Bigger models form induction sooner,** which leaves less plateau to skip.
- **The longest plateau** is web text with the 2L d256 model: about 5,000 flat steps, then slow formation, with no knee in the loss.
- **The only plateau visible in the training loss itself** is code with the 2L d512 model: loss is flat from about step 1,000 to 1,875, then drops 0.34 by step 2,450. We used this as the test bed.
- **Reference points on code:**
  - Uniform guessing: 9.01 nats.
  - A unigram model: 6.84.
  - A smoothed bigram model fitted on 60M training tokens: 4.67.
  - The 2L d512 model passes the bigram level at about step 200.

## Single edit at step 1,000 (code, 2L attention-only d512)

Session: `runs/pretrain_agent/code_d512_s0_step1000/`.
- **Prompt and tools:** `PROMPT.md` and `pretrain/agent_tools.py`. The agent may not use gradients or fitting; `tok.weight` and `head.weight` are out of scope; the edit is capped at 400k entries.
- **Arms:** all resume from `step_01000.pt` with the same optimizer state and the same batch sequence. The resumed control matches the original run to within 0.0005.

**The edit** (17.7 min, 173k tokens, 184 forward calls), 295,328 entries:

| Tensor | Entries | Change norm | Purpose |
|---|---|---|---|
| `pos.weight` | 32,736 | 8.96 | sinusoids at 0.5–1.5 rad/token in the 64 lowest-variance dimensions |
| `blocks.0.ln1.weight` | 448 | 1.58 | keeps the other heads' input scale unchanged |
| `blocks.0.attn.qkv.weight` | 98,304 | 23.9 | L0H6 becomes a previous-token head (Q is the sinusoids rotated by one step); V carries a 64-dim token code |
| `blocks.0.attn.proj.weight` | 32,768 | 4.64 | L0H6 writes that code into quiet residual dimensions |
| `blocks.1.attn.qkv.weight` | 98,304 | 60.2 | L1H2 becomes an induction head (K reads the previous-token code, Q and V read the current token's code) |
| `blocks.1.attn.proj.weight` | 32,768 | 4.38 | L1H2's output maps code to unembedding (Hebbian) |

**Immediate effect on training windows:**
- Loss: 3.399 → 3.544.
- Copy benefit: 0.081 → 0.122.
- L0H6 previous-token score: 0.025 → 0.815.
- L1H2 induction score: 0 → 0.14.
- The random edit of the same size raised the loss to 6.07.

**Continued training (held-out val loss):**

| Arm | 1,250 | 1,500 | 2,000 | 3,000 | 5,000 | 10,000 | 15,000 | Mean over 1k–15k |
|---|---|---|---|---|---|---|---|---|
| No edit | 3.000 | 2.939 | 2.669 | 2.351 | 2.160 | 1.987 | 1.911 | 2.1359 |
| Agent | 2.749 | 2.619 | 2.464 | 2.320 | 2.153 | 1.978 | 1.905 | 2.1039 |
| Random | 3.064 | 2.963 | 2.739 | 2.360 | 2.170 | 1.999 | 1.919 | 2.1531 |

**Mean gap to no edit, by window:**

| Steps | Agent − none | Random − none |
|---|---|---|
| 1,000–2,500 | −0.207 | +0.099 |
| 2,500–5,000 | −0.022 | +0.011 |
| 5,000–10,000 | −0.007 | +0.008 |
| 10,000–15,000 | −0.010 | +0.005 |
| last 1,000 | −0.007 | +0.007 |

- **Steps saved:** how many steps later the control reaches the agent's loss. 650 at step 1,250, 400 at 2,000, 200 at 3,000 and 100 at 5,000. Later values aren't meaningful because the curve is flat.
- **Copy benefit stays higher in the agent arm** to the end: 1.28 vs 1.19. The random arm ends at 1.22.

## Follow-up sessions at step 3,000 (from the agent arm)

| Session | Objective and tools | Time, tokens | Calls | Submission |
|---|---|---|---|---|
| `code_d512_s0_multi/s2_step3000` | same rules as step 1,000; loss of the continued run | 5.5 min, 102k | 42 | **empty edit** |
| `code_d512_s0_multi/s2b_step3000_deep` | scored on the loss at step 15,000; immediate cost allowed; `T.lookahead` (10 paired training runs of up to 1,500 steps, different batch order, training windows only) | 21 min, 186k | 78 + 3 look-aheads | **empty edit** |

**Look-aheads in the deep session** (training-window loss at step 4,500; paired, same batches):

| Candidate | Loss at 4,500 | Gap to no edit |
|---|---|---|
| No edit | 2.149 | – |
| A: L0H5 rebuilt as a second token-code head (complementary 64-dim code, output weights zero) | 2.160 | +0.064 at the start, then flat at +0.011 to +0.016 over the last 800 steps |
| C: function-preserving QK rescale (Q×3, K÷3) on 6 L1 heads | 2.152 | −0.001 to +0.003 (noise) |

**What the two sessions found:**
- **The planted circuit is intact:** L0H6 previous-token score 0.997, L1H2 induction score 0.46.
- **Copying is close to its limit.** Confidence matches accuracy, the model prefers longer matches, and a perfect copy mechanism through the same attention made loss worse. A flat bonus for every identifier already in the window would gain at most 0.016.
- **The 2 L1 heads read the current token mostly through one 64-dim code,** L0H4's output (77% of L1H2's value input). Adding a second code channel cost loss for as long as it was measured.
- **No head is cheap to remove:** zeroing any one costs at least about 0.055.
- **Indentation:** an ideal tracker is worth at most about 0.03, and L1H1 is already learning one.

## Head usage of the trained control (step 15,000)

Script: `runs/pretrain_agent/head_usage.py`, 48 validation windows, base loss 2.105.
- **Removal cost of each head** (zeroing its output):
  - L0: 0.09–1.76. The current-token head, L0H4 (mean distance 2.7, 96% of its attention within 4 tokens), costs 1.76.
  - L1: 0.10–0.23. The long-range heads, L1H0 and L1H5 (mean distance 116–147), cost about 0.21–0.23.
- **Redundancy:** joint removal of a pair costs a median **1.08×** the sum of the single costs, so heads are slightly complementary. The most overlapping pairs are L0 H0+H3 (0.78×) and L0 H4+H5 (0.85×).

## Memorization vs generalization (code models, step 15,000)

Script: `pretrain/probe_understanding.py`.

| Probe | 6L GPT d384 (17.1M) | 2L attention-only d512 (10.8M) |
|---|---|---|
| Loss, training vs unseen files | 1.580 vs 1.547 | 1.906 vs 1.868 |
| Greedy continuation of 32 tokens: share of tokens right (exact continuations), training vs unseen files | 15% (2%) vs 20% (5%) | 13% (2%) vs 16% (4%) |
| Made-up identifiers: loss per name, first vs later uses | 32.8 vs **3.0** nats | 33.2 vs 5.8 nats |
| Random nested `([{` brackets: correct closer ranked first (chance 33%) | **99%** | 94% (83% at depth 4) |
| Indentation after `if`/`def`/`for`/`while`/`class` lines | 100% | 99% |

- **No memorization signal:** these models generalize Python syntax, idioms and name binding.
- **The samples don't understand what code does.** `fibonacci(n)` returns `n`, a file opened for reading gets written to, there are undefined variables, and lines repeat.
- **A pitfall:** a prompt ending in a bare `\n` token pushes these tokenizers into a rare state, because in training a newline and the next line's indentation form one token. End prompts at the colon instead.

## Cost and disk

- **Agent sessions:** 3, totalling about 45 minutes and 460k tokens.
- **GPU:** about 6 hours in all, including the grid.
- **Disk:**
  - Data: `pretrain/data*`, about 3 GB.
  - Runs: `runs/pretrain*`, about 34 GB, mostly checkpoints with optimizer state.
  - Keep `runs/pretrain_code/attn2L_d512_s0/ckpt/step_01000.pt` and `runs/pretrain_agent/code_d512_s0_step1000/arm_agent/ckpt/step_03000.pt`. They are the starting points of all the sessions.

## Not done

- More seeds. Without them, the remaining 0.006 gap at step 15,000 can't be told apart from noise.
- A hand-written-circuit arm.
- The planned step-6,000 and step-10,000 interventions, dropped after two empty edits at step 3,000.
- The same pipeline on the 6L GPT with MLPs, which has more structure that could form late or badly.

# Results: would plain RL have fixed Sparks' reasoning loop? (2026-11-04)

Sparks' Experiment 8 fixed thinking-mode reasoning loops in Qwen3.5-0.8B by editing 0.016% of its weights, with no training afterwards. The paper never tested the counterfactual: would ordinary training have fixed the same failure? This run tests that, with no edit arm.

## Setup

- **Model:** base Qwen3.5-0.8B in thinking mode, 4,096-token limit. At this limit it reproduces Sparks' failure (`rl/README.md`): 99% of answers hit the limit and 61% are verbatim loops.
- **Training:** `rl/grpo.py --thinking --max-new 4096 --temperature 0.6 --trunc-zero --gen-batch 32 --micro 2`, 30 steps, seed 0.
  - GSM8K train, 8 problems × 8 samples per step, lr 1e-6, no weight decay. These are the same settings as the earlier RL arms.
  - Only text after `</think>` is scored. With `--trunc-zero`, an answer that runs to the limit gets 0, so only finished answers earn reward.
  - Sampling at T = 0.6 without top-k or top-p, so the training log-probs match the sampling distribution. Sparks' settings add top-k 20 and top-p 0.95.
- **Held-out test:** `rl/eval.py --suites thinkprobe` on the step-30 checkpoint. The first 128 GSM8K test problems, thinking mode, 4,096 tokens, Sparks' sampling (T = 0.6, top-k 20, top-p 0.95). This is the test that gave the base numbers.
- **Prediction before running:** a stall, since a near-zero success rate should give near-zero gradient. Decision rule: more than 25% of training answers finish thinking by step 30 means training escapes; staying at about the starting 5–10% means a stall.

## Held-out test

| | Accuracy | Loops | Hit the limit | Boxed answer | Average length |
|---|---|---|---|---|---|
| Base model | 3.9% | 61% | 99% | 36% | 4,071 |
| **After 30 GRPO steps** | **63.3%** | **3.9%** | **3.9%** | **100%** | **2,056** |

## Training samples, averaged over blocks of steps

| Steps | Finish thinking | Loops | Accuracy | Average length | Groups with no signal (of 8) |
|---|---|---|---|---|---|
| 1–2 | 4% | 39% | 3% | 4,011 | 5.5 |
| 3–5 | 30% | 21% | 25% | 3,381 | 4.7 |
| 6–10 | 47% | 12% | 36% | 2,966 | 2.8 |
| 11–15 | 77% | 6% | 53% | 2,350 | 2.2 |
| 16–20 | 85% | 6% | 60% | 2,175 | 2.6 |
| 21–25 | 94% | 3% | 72% | 2,005 | 3.0 |
| 26–30 | 94% | 1% | 63% | 2,075 | 2.8 |

Single steps swing widely: accuracy ranged from 44% to 81% over steps 26–30, because each step uses different problems.

## Findings

- **Training escapes, fast.** The decision rule was met at step 4 (30% finish thinking). The loop is mostly gone by step 13, about 1.6 hours in.
- **The prediction was wrong.** Without top-k and top-p, about 5% of answers at step 1 finish thinking, against 1 in 128 under Sparks' sampling. That gave 2–3 groups per step with a learning signal, which was enough. The failure is not a zero-gradient trap.
- **RL does more than remove the symptom.** Held-out accuracy rose from 3.9% to 63.3%, and every answer gives a boxed result.

## Cost

- **GPU:** 220 minutes of training (about 400 s of sampling and 40 s of training per step) and 14.5 minutes for the held-out test, on the RTX 4070. Sampling time didn't fall as answers shortened, because each generation batch runs until its longest sample stops.
- **One crash before step 1:** generating all 64 samples at once ran out of GPU memory. Batches of 32 fit, at 6.3 GiB peak.

## Not done

- **Side effects.** IFEval and non-thinking GSM-Symbolic on the step-30 checkpoint, against the base results in `runs/eval/base/`. This is the one place an edit that touches 0.016% of the weights could still beat RL.
- **More seeds,** a 32k-token limit like Sparks, and training sampled with top-k and top-p.
- **An edit arm.** We didn't reproduce Sparks' edit, so there's no head-to-head comparison on cost or accuracy.

## Files

- **Code:** `rl/grpo.py`, with new `--thinking` and `--gen-batch` flags, and finish-thinking and loop rates logged per step.
- **Training run:** `runs/grpo/thinkloop_s0/` (`log.jsonl`, `rollouts/`, `ckpt/step_00030/`), with output in `runs/grpo/thinkloop_s0.out`.
- **Test run:** `runs/eval/thinkloop_s0_step30/` (`summary.json`, `thinkprobe_think.jsonl.gz`).
- **One-step smoke test:** `runs/grpo/thinkloop_smoke/`.


# Results: training-free repair of spurious features in ResNet-50 (Experiment 2, 2026-11-04)

Plan, pass bar and predictions (fixed before any arm ran): [PLAN_2026-11-04_spurious_repair.md](PLAN_2026-11-04_spurious_repair.md). Code: `vision_spurious/`. Runs: `runs/vision_spurious/`.

## Setup

- **Model:** torchvision ResNet-50, `IMAGENET1K_V1`, frozen. The edit is final.
- **Failure:** Spurious ImageNet (Neuhaus et al., ICCV 2023): 100 classes × 75 photos that show only a class's spurious cue.
- **Clean set:** ImageNetV2 matched-frequency.

**Splits, seed 0.** The agents and every fitted or scripted baseline see only dev:
- Spurious images of 50 "dev classes", 25 per class (1,250).
- 3 ImageNetV2 photos per class (3,000).

Eval is held out:
- 50 new spurious images per dev class (2,500).
- 7 ImageNetV2 photos per class (7,000).
- All 75 images of each of the other 50 "unseen" classes (3,750).

**Metrics:**
- **Hit:** the share of spurious images predicted top-1 as their class.
- **AP:** the benchmark's spurious score. It is the average precision of p_k, ranking clean photos of k above spurious photos of k. Here it has only 7 positives per class, so it isn't comparable with the paper's table.
- **Top-1:** on the 7,000 ImageNetV2 eval photos.

**Agents:** three blind Opus 5.5 sessions. Each could edit up to 200,000 entries in any float tensor, had 300,000 forward images and about 30 minutes, and could use no gradients or fitting.

## Main table (held-out eval)

| Arm | Entries changed | Dev hit | Dev AP | Dev ROC AUC | Top-1 | Δ top-1 (paired SE) | Top-1, dev classes (350) | Unseen hit | Unseen AP |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 50.2% | 0.476 | 0.621 | 62.91% | – | 62.3% | 45.1% | 0.554 |
| **agent s1** | 3,169 | 21.8% | 0.570 | 0.717 | 62.79% | −0.13 (0.07) | 57.1% | 47.1% | 0.548 |
| **agent s2** | 5,000 | **15.7%** | **0.670** | **0.801** | 62.34% | −0.57 (0.13) | 59.7% | 46.9% | 0.538 |
| **agent s3** | 29,912 | 17.2% | 0.594 | 0.742 | 62.77% | −0.14 (0.09) | 58.9% | 47.5% | 0.545 |
| random, same entries s1 / s2 / s3 | same as agent | 48.4 / 48.3 / 48.5% | 0.477 / 0.476 / 0.473 | 0.62–0.63 | 62.80–62.94% | −0.11 to +0.03 | 62.6 / 60.6 / 62.9% | 43.6–45.4% | 0.553–0.557 |
| random, same count s1 / s2 / s3 | same as agent | 49.9 / 49.7 / 49.6% | 0.472–0.475 | 0.62 | 62.79–63.03% | −0.11 to +0.13 | 61.7 / 61.7 / 62.9% | 43.3–44.6% | 0.553–0.554 |
| proj (scripted fc projection) | 102,450 | 0.2% | 0.772 | 0.889 | 61.49% | −1.43 (0.18) | **24.6%** | 48.1% | 0.553 |
| spufix_like (SpuFix clamp, wrapper) | 0 | 0.1% | 0.644 | 0.768 | 60.99% | −1.93 (0.20) | 13.7% | 48.1% | 0.553 |
| llr (last-layer retraining), trained to ≤2% dev hit | 2,049,000 | 5.4% | 0.498 | 0.648 | 61.04% | −1.89 (0.38) | 38.6% | 43.7% | 0.554 |
| ft_l4 (layer4 + fc fine-tune), trained to ≤2% dev hit | 17,013,698 | 4.6% | 0.619 | 0.759 | 61.94% | −0.99 (0.34) | 50.3% | **35.9%** | 0.570 |

The paired SE is over the 7,000 clean eval photos. The dev-class top-1 is measured on 350 photos, so its unpaired SE is about 2.6 points.

**Note.** The llr and ft_l4 rows sit at an extreme operating point, because they were trained until the dev hit rate was at most 2%. The fair comparison at the agents' own operating points is in "Amendment A1" below.

**Per class** (50 dev classes; AP counts as up or down when it moves by more than 0.01; top-1 is measured on each class's 7 clean photos):

| Arm | AP up | AP down | Class-k top-1 down | Class-k top-1 up |
|---|---|---|---|---|
| s1 | 45 | 0 | 15 | 0 |
| s2 | 48 | 0 | 14 | 13 |
| s3 | 48 | 1 | 13 | 6 |
| proj | 48 | 2 | 45 | 0 |
| ft_l4 | 45 | 3 | 31 | 3 |

## Pass bar

The thresholds come from the base model:
1. **Fixes the failure:** dev hit ≤ 25.1% and dev AP ≥ 0.576.
2. **No collateral damage:** top-1 ≥ 62.41% and dev-class top-1 ≥ 59.29%.
3. **Not just any perturbation:** the matched same-entries random edit fails condition 1.

| Session | 1: fixes the failure | 2: no collateral damage | 3: random control fails 1 | Pass |
|---|---|---|---|---|
| s1 | no: AP 0.570 is 0.006 short | no: dev-class 57.1%, 2.2 points short | yes | **no** |
| s2 | yes | no: top-1 62.34%, **0.07 points (5 photos) short**; dev-class OK | yes | **no** |
| s3 | yes | no: dev-class 58.9%, **0.4 points (about 1–2 photos of 350) short**; overall OK | yes | **no** |

**0 of 3 sessions pass, so the experiment fails its pass bar.** Two sessions miss by margins far below the metrics' noise. No baseline meets condition 2 either; they miss by 1–2 points overall and 12–49 points on the dev classes.

## Agents' own predictions against the outcome

| Session | Edit | Predicted | Held-out outcome |
|---|---|---|---|
| s1 | `fc.weight` only, 3,169 entries (‖Δ‖ = 2.04). Per dev class: the top 32 cue channels ×0.25, plus up to 32 object channels boosted to restore the clean-k logit. | hit 0.20–0.26, AP +0.05–0.08, top-1 ±0.2, dev-class −3 to −6 | hit 0.218 ✓, AP +0.094 (better), top-1 −0.13 ✓, dev-class −5.1 ✓ |
| s2 | `fc.weight` only, 5,000 entries (‖Δ‖ = 4.27). Per dev class: the top 50 cue channels zeroed and the top 50 object channels ×1.5. | hit 0.17–0.25, AP 0.68–0.72, top-1 −0.3 to +0.2, dev-class −1 to −3 | hit 0.157 (better), AP 0.670 (just below), **top-1 −0.57 (worse)**, dev-class −2.6 ✓ |
| s3 | `fc.weight` 29,862 + `fc.bias` 50. A graded cut of cue channels (by how much more they fire on cue-only images) and a graded boost of object channels; half of the clean-k logit change is restored through the bias. | hit 0.18–0.25, AP 0.58–0.62, top-1 −0.3 to +0.2, dev-class −3 to −6 | hit 0.172 (slightly better), AP 0.594 ✓, top-1 −0.14 ✓, dev-class −3.4 ✓ |

**10 of the 12 predicted intervals contain the outcome or err on the good side.** The two misses:
- s2's top-1: −0.57 against a predicted −0.3 to +0.2. Its ×1.5 object-channel boost lets dev classes steal predictions from other classes: top-1 on the 900 classes outside the benchmark fell from 62.68% to 62.21%.
- s2's AP: 0.670, just below its predicted 0.68–0.72.

Every session estimated dev-class damage with cross-validation on its 3 clean photos per class, and those estimates held. The in-sample dev-class numbers (s2: 72.7%, s3: 66.0%) were optimistic, as the agents themselves wrote.

## Findings

1. **All three sessions found the same mechanism and built the same kind of edit, independently.**
   - They cached penultimate features and defined "cue channels" for class k: channels with w_kj > 0 that fire much more on k's spurious images than on k's clean photos.
   - They checked that these channels are context detectors. The clean images that fire them most are sharks and coral for snorkel; envelopes and menus for quill and fountain pen; daisies for hummingbird and bee; skis for snowplow.
   - They then shrank those weights in fc row k and boosted the "object" channels to compensate.
   - All three edits are fc-only and change 0.01–0.12% of the model.
2. **The agents' edits dominate every baseline on the trade-off between the fix and the damage it causes.**
   - s2 reaches a higher dev AP than the layer4 fine-tune (0.670 against 0.619), with about half its top-1 loss (−0.57 against −0.99), about a fifth of its dev-class loss (−2.6 against −12.0 points), and 3,400× fewer changed entries.
   - The scripted projection gets the best AP (0.772) only by wiping out the classes themselves: dev-class top-1 falls to 24.6%. AP alone is gameable by suppressing class k globally, which is why the plan pairs it with dev-class top-1.
3. **Training on the same small dev set is worse than editing.** This holds at the agents' own operating points too: see Amendment A1 below.
   - Last-layer retraining hit its dev stopping rule (dev hit ≤ 2%) after 62 Adam steps, but held-out AP barely moved (0.476 → 0.498). It mostly learned to suppress class k (dev-class top-1 −24 points) and fit the 3 clean photos per class (dev clean top-1 64% → 83%).
   - The layer4 fine-tune generalizes better, but costs 1 point overall and 12 points on the dev classes.
4. **Random edits do nothing.** Random edits of the same size, at the same entries or at random positions, leave hit and AP within ±2 points and ±0.004.
5. **Unseen classes: no transfer, and slight harm.**
   - Each agent edit raises the unseen-class hit rate by 1.8–2.4 points. Random edits move it by −1.8 to +0.3, so this is at the edge of noise.
   - A likely reason: suppressing dev classes removes competitors for the unseen classes' spurious images. proj shows the same effect (+3.0).
   - Only ft_l4, which changes layer4 features, reduces the unseen hit rate (45.1% → 35.9%).
6. **The pass bar was strict relative to the noise.** The plan used 0.5 points for overall top-1, where the paired SE is 0.1–0.13. It used 3 points for dev-class top-1, measured on 350 photos with SE about 2.6. s2 and s3 miss by 5 photos and about 1–2 photos.

### The plan's predictions against the outcome

| Arm | Predicted | Outcome |
|---|---|---|
| Base | top-1 about 63%, AP 0.4–0.55 | correct (62.9%, 0.476) |
| Base hit rate | 30–45% | higher: 50% |
| proj | dev-class cost 1–3 points | wrong: −38 points |
| llr | the largest fix, with top-1 unchanged | wrong: no AP gain and −1.9 top-1 |
| ft_l4 | −1 to −3 points of top-1 | correct (−1.0) |
| Agents | 2 of 3 pass, AP comparable to proj | 0 of 3 pass by thin margins, lower AP than proj but far less damage |

## Amendment A1 (post-hoc): training stopped at the agents' operating point

**Why.** The original llr and ft_l4 arms were trained until the in-sample dev hit rate was at most 2%. The agents instead compared doses and stopped at an in-sample dev hit rate of 0.234 (s1), 0.165 (s2) and 0.173 (s3). The question here: **would plain training on the same images, stopped at the same point, have done the agents' job?**

**Setup** (`vision_spurious/train_curves.py`; the amendment is in the plan). Five variants, each started from the pretrained weights and trained on exactly the agents' dev data, with the loss CE(clean) − log(1 − p_k)(spurious):

| Variant | Trains | Optimizer | Extra |
|---|---|---|---|
| `llr` | fc | full-batch Adam, lr 1e-4 | – |
| `llr_rows` | only the 50 dev-class rows of fc, the agents' scope | full-batch Adam, lr 1e-4 | – |
| `llr_gentle` | fc | Adam, lr 2e-5 | + KL(original ‖ new) on all 3,000 clean dev images |
| `ft` | layer4 + fc | Adam, lr 1e-5, batch 64 | – |
| `ft_gentle` | layer4 + fc | Adam, lr 1e-5, batch 64 | + KL on each batch's clean images |

**Checkpoint selection used dev only:**
- **match sN:** the first step whose in-sample dev hit rate falls to agent sN's.
- **3-fold CV:** the lowest CV hit rate among steps that stay within the pass bar's damage limits on held-out dev folds.

**"Train s"** below is pure GPU training time, excluding feature caching and evaluation.

| Arm | Step | Train s | Dev hit (in-sample) | Held-out hit | Held-out AP | Δ top-1 | Top-1, dev classes | Unseen hit | Conditions 1 / 2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | – | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| **agent s1** | – | – | 0.234 | 21.8% | **0.570** | −0.13 | **57.1%** | 47.1% | no / no |
| llr @ match s1 | 11 | 0.07 | 0.230 | 23.5% | 0.476 | −0.60 | 50.0% | 45.5% | no / no |
| llr_rows @ match s1 | 21 | 0.11 | 0.231 | 23.4% | 0.471 | −0.29 | 49.1% | 47.7% | no / no |
| llr_gentle @ match s1 | 65 | 0.39 | 0.231 | 23.9% | 0.481 | −0.04 | 51.1% | 46.3% | no / no |
| ft @ match s1 | 28 | 0.87 | 0.223 | 23.4% | 0.544 | −0.90 | 52.6% | 41.1% | no / no |
| ft_gentle @ match s1 | 30 | 0.97 | 0.230 | 24.5% | 0.546 | −0.23 | 54.3% | 44.0% | no / no |
| **agent s2** | – | – | 0.165 | **15.7%** | **0.670** | −0.57 | **59.7%** | 46.9% | yes / no |
| llr @ match s2 | 15 | 0.09 | 0.163 | 17.3% | 0.478 | −0.81 | 48.0% | 44.9% | no / no |
| llr_rows @ match s2 | 29 | 0.15 | 0.162 | 17.2% | 0.476 | −0.37 | 46.3% | 48.0% | no / no |
| llr_gentle @ match s2 | 94 | 0.56 | 0.165 | 18.3% | 0.485 | −0.31 | 48.6% | 46.4% | no / no |
| ft @ match s2 | 42 | 1.30 | 0.165 | 20.3% | 0.555 | −0.74 | 53.1% | 41.1% | no / no |
| ft_gentle @ match s2 | 50 | 1.62 | 0.164 | 20.4% | 0.562 | −0.41 | 55.1% | 43.2% | no / no |
| **agent s3** | – | – | 0.173 | **17.2%** | **0.594** | −0.14 | **58.9%** | 47.5% | yes / no |
| llr @ match s3 | 15 | 0.09 | 0.163 | 17.3% | 0.478 | −0.81 | 48.0% | 44.9% | no / no |
| llr_rows @ match s3 | 28 | 0.14 | 0.168 | 17.6% | 0.472 | −0.34 | 46.9% | 48.0% | no / no |
| llr_gentle @ match s3 | 90 | 0.54 | 0.171 | 18.7% | 0.485 | −0.30 | 48.9% | 46.4% | no / no |
| ft @ match s3 | 40 | 1.24 | 0.171 | 20.4% | 0.553 | −0.70 | 53.1% | 41.2% | no / no |
| ft_gentle @ match s3 | 48 | 1.55 | 0.168 | 20.8% | 0.561 | −0.36 | 55.1% | 43.0% | no / no |
| llr @ 3-fold CV | 2 | 0.03 | 0.469 | 44.8% | 0.475 | +0.24 | 60.3% | 45.5% | no / yes |
| llr_rows @ 3-fold CV | 4 | 0.02 | 0.463 | 44.2% | 0.473 | +0.03 | 60.0% | 46.0% | no / yes |
| llr_gentle @ 3-fold CV | 15 | 0.09 | 0.434 | 42.6% | 0.475 | +0.20 | 59.4% | 45.6% | no / yes |
| ft @ 3-fold CV | 6 | 0.19 | 0.415 | 40.7% | 0.490 | −0.43 | 56.9% | 43.7% | no / no |
| ft_gentle @ 3-fold CV | 8 | 0.26 | 0.387 | 38.8% | 0.497 | −0.63 | 56.3% | 44.1% | no / no |

**Oracle check** (diagnostic only; it selects with held-out data, which the rules forbid). Across the whole held-out trajectory of every variant:
- No training checkpoint meets conditions 1 and 2 together.
- The best dev-class top-1 at a held-out hit rate of 25% or less is 55.7% (`ft_gentle`, step 60), against 57.1–59.7% for the agents.
- The highest AP anywhere is 0.609 (`ft`, at 6.8% hit and 51.7% dev-class top-1), against the agents' 0.570–0.670.
- Plot: `runs/vision_spurious/tradeoff.png`. Its x-axis is held-out hit; its y-axes are dev-class top-1 and overall top-1. The agents sit 2–5 points above the best training curve (`ft_gentle`) and about 10 points above the fc curves on dev-class top-1 at equal hit rate.

**Findings of the amendment:**
1. **At equal hit rates, the agents' edits beat plain training on what matters for the reported classes.**
   - **Spurious AP:** 0.57–0.67 for the agents, against 0.47–0.49 for fc training and 0.54–0.56 for layer4 training.
   - **Dev-class top-1:** 57–60% for the agents, against 46–51% for fc training and 53–55% for layer4 training.
   - **Why:** fc training reaches the agents' hit rate mostly by lowering class k everywhere, so AP doesn't move. The agents removed specific context channels and compensated through object channels.
2. **On overall top-1, gentle training is as good as the careful agents.** `llr_gentle` costs −0.04 to −0.31 and `ft_gentle` −0.23 to −0.41, against s1 and s3 at −0.13/−0.14. Both gentle variants beat s2's −0.57. The edits' advantage is concentrated on the reported classes, not on the other 950.
3. **Only layer4 training transfers to unseen classes.** It cuts their hit rate by 1–4 points (41–44% against 45.1%). fc training and the agents' fc edits don't transfer, and slightly raise it.
4. **Selecting a training checkpoint by 3-fold CV on dev barely trains at all** (steps 2–15, held-out hit 39–45%). With 3 clean photos per class, the damage limits bind almost immediately: training can't find a point that both fixes the failure and passes the damage check on dev. The agents did find one, using the same 3 photos, and their CV estimates held on eval.
5. **Training is far cheaper.** The matched points take 11–94 steps and 0.07–1.6 s of GPU training. That is about 2 minutes end to end, including a one-off feature pass over dev, against about 4–5 minutes and many tokens per agent session. But the cheap result is worse.
6. **The pass bar is unchanged.** No training checkpoint passes condition 2 at any operating point that meets condition 1. The agents remain 0 of 3, by thin margins.

## Amendment A2 (post-hoc): training with ImageNet-train data

**Why.** A1 trained on 3 clean photos per class. A2 gives training a realistic amount of labelled data, as every earlier experiment's training had.

**Data.** The user accepted the ImageNet terms and downloaded 12 `ILSVRC/imagenet-1k` train shards (5.6 GB). `vision_spurious/imagenet_pool.py` read them as data only (`python -I`) and kept up to 50 photos per class, preprocessed like eval:
- **Pool size:** 48,092 photos. Per class: minimum 23, 5th percentile 41, median 50. Dev classes: minimum 39, median 50.
- **Pool-val:** the first min(10, n/5) per class are held out (9,445 photos). The rest form pool-train (38,647).

**Training** (`vision_spurious/train_full.py`).
- **Positives:** pool-train plus the 3 ImageNetV2 dev photos per class.
- **Negatives:** the same 25 dev cue-only images per class.
- **Loss:** the A1 form, with each term averaged over its own images.
- **Variants:**
  - `ft_gentle_in` and `ft_in`: layer4 + fc, lr 1e-5, 64 positives + 16 negatives per step, at most 400 steps; the first with KL to the original model.
  - `llr_gentle_in`: fc, full batch, lr 2e-5, + KL.
  - `llr_rows_in`: the 50 dev-class fc rows only, full batch, lr 1e-4.

**Stopping, without eval data:**
- **match sN:** as in A1.
- **practitioner rule:**
  - Three runs, each trained on 2/3 of every class's cue images.
  - Choose the step with the lowest CV cue hit rate, among steps where pool-val top-1 is no more than 0.5 points below base and pool-val dev-class top-1 no more than 3 points below.
  - Score the full run at that step.

| Arm | Step | Train s | Dev hit (in-sample) | Held-out hit | Held-out AP | Δ top-1 | Top-1, dev classes | Unseen hit | Conditions 1 / 2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | – | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| **agent s1** | – | – | 0.234 | 21.8% | **0.570** | −0.13 | **57.1%** | 47.1% | no / no |
| ft_gentle_in @ match s1 | 20 | 2.1 | 0.231 | 25.2% | 0.522 | −1.10 | 53.4% | 40.5% | no / no |
| ft_in @ match s1 | 20 | 2.1 | 0.222 | 23.5% | 0.521 | −1.60 | 51.4% | 39.0% | no / no |
| llr_gentle_in @ match s1 | 60 | 2.4 | 0.214 | 22.4% | 0.477 | −0.07 | 49.1% | 47.1% | no / no |
| llr_rows_in @ match s1 | 21 | 0.6 | 0.225 | 23.1% | 0.468 | −0.26 | 49.1% | 47.7% | no / no |
| **agent s2** | – | – | 0.165 | 15.7% | **0.670** | −0.57 | **59.7%** | 46.9% | yes / no |
| ft_gentle_in @ match s2 | 40 | 4.3 | 0.142 | 16.4% | 0.544 | −0.84 | 53.4% | 39.3% | no / no |
| ft_in @ match s2 | 40 | 4.2 | 0.123 | 13.6% | 0.543 | −1.34 | 50.6% | 36.6% | no / no |
| llr_gentle_in @ match s2 | 80 | 3.1 | 0.164 | 18.0% | 0.479 | +0.01 | 48.6% | 47.3% | no / no |
| llr_rows_in @ match s2 | 28 | 0.8 | 0.162 | 17.3% | 0.468 | −0.31 | 47.1% | 48.0% | no / no |
| **agent s3** | – | – | 0.173 | 17.2% | **0.594** | −0.14 | **58.9%** | 47.5% | yes / no |
| ft_gentle_in @ match s3 | 30 | 3.2 | 0.172 | 19.4% | 0.534 | −0.94 | 51.7% | 39.3% | no / no |
| ft_in @ match s3 | 30 | 3.2 | 0.166 | 17.7% | 0.536 | −1.67 | 50.0% | 37.1% | no / no |
| llr_gentle_in @ match s3 | 80 | 3.1 | 0.164 | 18.0% | 0.479 | +0.01 | 48.6% | 47.3% | no / no |
| llr_rows_in @ match s3 | 27 | 0.7 | 0.170 | 18.3% | 0.467 | −0.29 | 47.7% | 47.9% | no / no |
| ft_gentle_in @ practitioner rule | 0 (no step qualifies) | 0 | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| ft_in @ practitioner rule | 0 (no step qualifies) | 0 | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| llr_gentle_in @ practitioner rule | 17 | 0.7 | 0.416 | 41.0% | 0.471 | +0.11 | 58.6% | 45.8% | no / no |
| llr_rows_in @ practitioner rule | 5 | 0.1 | 0.438 | 43.0% | 0.473 | +0.01 | 59.4% | 46.3% | no / yes |

**Oracle check** (diagnostic only; it selects with held-out data). Across all four held-out trajectories:
- No checkpoint meets conditions 1 and 2 together.
- The best dev-class top-1 at a held-out hit rate of 25% or less is 54.3% (`ft_gentle_in`, step 90).
- The highest AP is 0.608 (`ft_gentle_in`, step 200), at 7.2% hit, 52.9% dev-class top-1 and −0.50 overall.
- Plot: `runs/vision_spurious/tradeoff_imagenet.png`.

**Findings of A2:**
1. **About 14× more clean data doesn't change the picture.** At each agent's hit rate, training gets:
   - **AP:** 0.47–0.54, against the agents' 0.57–0.67.
   - **Dev-class top-1:** 47–53%, against 57–60%.

   fc training still mostly lowers the class everywhere. Layer4 training with ImageNet data also costs 0.8–1.7 points of overall top-1 at the matched points. That is more than with dev data only (A1), and it recovers only partly with further steps.
2. **Gentle fc training with ImageNet data now ties or slightly beats every agent on overall top-1** (+0.01 to −0.07), but still loses about 10 points on the repaired classes.
3. **The practitioner rule declines to train the layer4 models at all.** For layer4 + fc, no step keeps pool-val within the damage limits, so the rule picks step 0. For fc, it stops at steps 5–17, with held-out hit still 41–43%. A practitioner following a standard validation rule would ship roughly the unfixed model.
4. **Layer4 training is the only route that transfers to the unseen classes,** cutting their hit rate from 45.1% to 37–41%. The agents' fc edits don't.
5. **Cost:** 20–80 steps and 1–4 s of GPU training to reach the matched points. A whole variant took 3–8 minutes, including the 3-fold practitioner runs and evaluation, plus about 3 minutes to build the pool and 3 minutes to cache features.

**Caveats:**
- **ResNet-50 was originally trained on these exact ImageNet-train images.** That is what a practitioner would have, so it's the right counterfactual. But pool-val is not new to the model: base pool-val top-1 is 87.6% against 62.9% on ImageNetV2. So pool-val understates the damage that held-out photos show, and the rule's damage limits are measured on a near-training distribution.
- **Steps were capped** at 400 for layer4 + fc and at 1,500–3,000 for fc, and all four variants reached every matched point.

## Amendment A3 (post-hoc): DFR-style retraining and a scripted version of the agents' recipe

Both arms are in `vision_spurious/a3_arms.py` and run on cached penultimate features. No eval data is used in any selection. The full design is in the plan.

**DFR-style balanced last-layer retraining** (`dfr_lam*`):
- **Start:** the pretrained fc, with an L2 penalty λ toward the original weights.
- **Loss:** three terms of equal weight:
  - CE on all pool-train photos and the dev photos
  - CE on the dev classes' pool photos, with cue-containing photos (cue score above the class median) oversampled 2×
  - −log(1 − p_k) on the 25 cue images per class
- **Grid:** λ ∈ {0, 0.01, 0.1, 1}, full-batch Adam lr 1e-4, 300 steps.
- **Selection:** the practitioner rule (3-fold cue CV plus pool-val damage limits) over all (λ, step) pairs picked **λ = 0.1, step 300**. For the matched points, each agent target takes the λ with the best pool-val dev-class top-1 at its first matched step.

**Scripted recipe** (`script_*`): the agents' shared recipe in closed form, with session 2's and session 1's settings. Clean means come either from the ImageNet pool (`_pool`, ~50 photos per class) or from the 3 dev photos (`_dev3`).

| Arm | Entries / step (GPU s) | Held-out hit | AP | Δ top-1 | Top-1, dev classes | Unseen hit | Conditions 1 / 2 |
|---|---|---|---|---|---|---|---|
| none | – | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| agent s1 | 3,169 | 21.8% | 0.570 | −0.13 | 57.1% | 47.1% | no / no |
| agent s2 | 5,000 | 15.7% | 0.670 | −0.57 | 59.7% | 46.9% | yes / no |
| agent s3 | 29,912 | 17.2% | 0.594 | −0.14 | 58.9% | 47.5% | yes / no |
| **script_s1_dev3** | 3,169 | 21.8% | 0.570 | −0.13 | 57.1% | 47.1% | no / no |
| **script_s2_dev3** | 5,000 | 15.7% | 0.670 | −0.57 | 59.7% | 46.9% | yes / no |
| script_s1_pool | 2,527 | 34.1% | 0.534 | +0.04 | 61.7% | 46.5% | no / yes |
| script_s2_pool | 4,792 | 29.1% | 0.653 | −0.76 | **70.0%** | 45.0% | no / no |
| dfr λ=0.01 @ match s1 | step 27 (0.6) | 26.4% | 0.540 | −0.49 | 60.3% | 43.5% | no / yes |
| dfr λ=0.01 @ match s2 | step 47 (1.1) | 22.8% | 0.565 | −0.40 | 62.6% | 43.3% | no / yes |
| dfr λ=0.1 @ match s1 | step 48 (1.1) | 26.3% | 0.549 | −0.39 | 60.3% | 45.8% | no / yes |
| dfr λ=0.1 @ match s3 | step 170 (4.0) | 22.0% | 0.577 | −0.24 | 62.6% | 45.4% | **yes / yes** |
| **dfr λ=0.1 @ practitioner rule** | step 300 (7.1) | 21.5% | 0.583 | −0.29 | **62.9%** | 45.2% | **yes / yes** |

**Other DFR settings:**
- **λ = 0:** the matched points have AP 0.536–0.567 and −0.6 to −0.7 overall.
- **λ = 1:** never reaches any agent's dev hit rate.
- **Oracle check** (held-out, diagnostic only): λ = 0.1 meets both pass-bar conditions at every checked step from 175 to 300, and λ = 0.01 at steps 75 and 125. With λ = 0 at step 300 the AP reaches 0.682 and dev-class top-1 68.0%, but overall top-1 drops 1.63 points.

**Findings of A3:**
1. **The agents' edits are exactly a closed-form recipe.** `script_s2_dev3` reproduces session 2's held-out numbers to the last digit. `script_s1_dev3` reproduces session 1's: 0.218 hit against 0.2184; everything else is identical. Session 3's graded variant is the same idea. Once the recipe is written down, no AI judgment is needed to apply it.
2. **The recipe with better clean statistics trades differently.** With pool means instead of 3 photos, fewer channels qualify as cue channels (4,792 entries changed against 5,000). Hit rates are higher (29–34%), but the repaired classes improve beyond the unfixed model: 70.0% top-1 for the s2 settings, against 62.3%. The s1 settings pass condition 2 but not condition 1. Neither passes.
3. **DFR-style balanced retraining, selected on dev only, is the first arm in this experiment to pass the pass bar.** It has 21.5% hit, AP 0.583, −0.29 overall and no loss on the repaired classes (62.9%). The margin is thin: AP clears 0.576 by 0.007. The practitioner rule picked the last step of the 300-step cap, and the held-out curve was still slowly improving.
   - Compared with the agents, DFR leaves more spurious hits than s2 and s3 (21.5% against 15.7–17.2%), and its AP is below s2's 0.670. But it does no damage to the repaired classes (62.9% against 57–60%).
   - The ingredients that matter, compared with A2's `llr_gentle_in`: balancing each class's real photos (cue-containing ones oversampled) against its cue images, and an L2 pull toward the original fc. With λ = 0, DFR fails condition 2 overall.
4. **DFR is cheap:** 7 s of GPU for the chosen run. The whole grid with CV took about 5 minutes. It needs about 40 labelled photos per class, and its oversampling uses the cue channels defined by the agents' recipe.
5. **No arm transfers to unseen classes except layer4 training.** DFR leaves them at 43–46%.

## Rule audit

- No session script uses `backward`, `autograd`, an optimizer, `lstsq` or `requires_grad`.
- No session script references `eval.pt`, `data_vision/derived`, the baseline scripts or other runs.
- **Forward budget used:** s1 8,500, s2 8,500 and s3 21,250 images, out of 300,000.
- **How the edits were evaluated:** all three agents evaluated their fc-only edits offline from cached penultimate features. That is exact for fc edits, and it is computing with the network, not fitting.
- **Variant comparisons:** hand-chosen dose ladders of 5–9 variants per session, with no automated search.

## Files

- **Code:**
  - `vision_spurious/` (`prepare_data.py`, `common.py`, `eval.py`, `agent_tools.py`, `session.py`, `baselines.py`, `summarize.py`)
  - `vision_spurious/prompt_template.md`
- **Eval outputs:**
  - `runs/vision_spurious/eval/<arm>.json`: summary plus per-class numbers
  - `runs/vision_spurious/eval/<arm>_logits.pt`: fp16 logits
  - `runs/vision_spurious/summary.txt`
- **Sessions:** `runs/vision_spurious/session_{1,2,3}/` (`PROMPT.md`, `work/notes.md`, scripts, `submitted_edit.pt`, `random0_{same,count}_edit.pt`)
- **Baseline weights and training logs:** `runs/vision_spurious/arms/`
- **Amendment A1:**
  - code: `vision_spurious/train_curves.py`, `vision_spurious/summarize_curves.py`, `vision_spurious/plot_tradeoff.py`
  - trajectories and selections: `runs/vision_spurious/curves/<variant>.json`
  - selected checkpoints: `runs/vision_spurious/eval/<variant>_{match_s1,match_s2,match_s3,cv}.json`
  - table: `runs/vision_spurious/summary_curves.txt`
  - plot: `runs/vision_spurious/tradeoff.png`
- **Amendment A2:**
  - code: `vision_spurious/imagenet_pool.py`, `vision_spurious/train_full.py`
  - pool: `data_vision/imagenet_pool/` (shards in `data_vision/imagenet_train_shards/data/`)
  - trajectories: `runs/vision_spurious/curves_in/`
  - selected checkpoints: `runs/vision_spurious/eval/<variant>_in_{match_sN,prac}.json`
  - table: `runs/vision_spurious/summary_curves_in.txt`
  - plot: `runs/vision_spurious/tradeoff_imagenet.png`
- **Amendment A3:**
  - code: `vision_spurious/a3_arms.py`
  - DFR trajectories: `runs/vision_spurious/curves_in/dfr_lam*.json`, `dfr_selection.json`
  - evals: `runs/vision_spurious/eval/{script_*,dfr_lam*}.json`
  - table: `runs/vision_spurious/summary_a3.txt`
  - plot: `tradeoff_imagenet.png`, updated with both arms

# Results: an agent's mid-training edit for ViT locality (2026-11-04)

Plan, pass bar and predictions: [PLAN_2026-11-04_vit_locality.md](PLAN_2026-11-04_vit_locality.md). Setup and scripts: `vision/README.md`.

## Setup

- **Model and recipe:** a ViT on CIFAR-10 with 2.69M parameters:
  - patch 4 (8×8 = 64 tokens), width 192, 6 blocks, 3 heads, learned position embeddings, mean pooling
  - AdamW (lr 1e-3, weight decay 0.05), 200-step warmup, cosine to 1e-5, batch 512
  - 4,000 steps (about 41 epochs), random crop and flip only
  - about 2–3 minutes per run on the RTX 4070
- **Edit point:** step 200 of 4,000 (5%), at about 35–40% test accuracy. Every arm resumes from the same checkpoint with the same optimizer state and the same batches. The `none` arm reproduces the base run's step-200 probe exactly.
- **Arms, seeds 0–2 each:**
  - `none`: no edit
  - `agent`: one blind Opus 5.5 session per seed, about 30 minutes, training images only, budget 900k entries
  - `random0`: same tensors, same number of changed entries and same change norm per tensor as that seed's agent edit, at random positions
  - `mimetic`: the scripted reference, mimetic attention (Trockman & Kolter 2023) added to every layer (`vision/mimetic.py`, 884,736 entries)
- **Recipe check, run before the arms:** two plain runs ended at 81.69% and 81.51%. A run with a fixed Gaussian locality bias in every head ended at 82.53%. So in this recipe, locality is worth only about +0.9 points.

## What each agent did, and what it predicted

| Seed | Edit | Entries | Immediate effect (training images) | Agent's prediction | Outcome: final Δ vs none |
|---|---|---|---|---|---|
| 0 | Clean x, y and x²+y² features written into `pos_embed` along low-variance residual directions. 3 of the 64 Q/K dimensions of heads 0–1 in blocks 0–3 rewritten so that the score is a Gaussian in distance (σ = 1–3 patches) | 21,552 | loss +0.041, accuracy −2.7 points | +1 to +2 points, "moderate-low" confidence | **−1.45** |
| 1 | The same idea: a 3-feature position code in directions the patch embedding can't write, plus Gaussian Q/K dimensions in all heads of blocks 0–2 (σ = 2.1 / 3.5 / 5.6). It chose the sharpest dose that was about loss-neutral | 22,710 | loss +0.003, accuracy ±0 | +0.5 to +2 points; about 60% to beat none | **+0.12** |
| 2 | 4 low-variance residual coordinates cleared in every block (all readers and writers zeroed) and filled with 2D sinusoidal position features. 4 Q/K dimensions of heads 0–1 in blocks 0–3 turned into a relative-position bias | 73,752 | loss +0.16, accuracy −4.6 points | +1.5 to +3 points; about 70% to beat none at the end, about 50% on the run average | **+1.03** |

The three agents reached the same diagnosis independently:
- At step 200 every head attends almost uniformly: mean distance is 3.7–4.4 patches, against 4.14 for uniform attention.
- `pos_embed` is weak.

They also built the same kind of fix. Each wrote a Gaussian locality prior into a few Q/K dimensions of the early heads and left the rest of each head untouched. Each checked dose–response and found that sharper locality cost more immediate loss.

## Test accuracy at step 4,000

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean | Δ vs none, per seed | Mean Δ: final / last-5 / span |
|---|---|---|---|---|---|---|
| none | 81.20 | 82.27 | 81.44 | 81.64 | | |
| agent | 79.75 | 82.39 | 82.47 | 81.54 | −1.45, +0.12, +1.03 | −0.10 / −0.12 / −0.34 |
| random0 | 81.10 | 81.90 | 81.32 | 81.44 | −0.10, −0.37, −0.12 | −0.20 / −0.11 / +0.10 |
| mimetic | 80.34 | 80.08 | 79.93 | 80.12 | −0.86, −2.19, −1.51 | −1.52 / −1.60 / −4.55 |

"Last-5" is the mean of the probes at steps 3,600–4,000. "Span" is the mean over steps 200–4,000.

## Head start: (arm − none) test accuracy by window, in points

The mean over seeds comes first, then seeds 0 / 1 / 2.

| Arm | Steps 300–500 | 600–1,000 | 1,100–2,000 | 2,100–3,000 | 3,100–4,000 |
|---|---|---|---|---|---|
| agent | +0.4 (−1.5, +1.6, +1.2) | −0.1 (−2.0, +0.6, +1.0) | −0.7 (−2.2, −0.2, +0.2) | −0.3 (−1.7, +0.1, +0.7) | −0.1 (−1.3, +0.2, +0.7) |
| random0 | +0.9 (+0.2, +1.9, +0.7) | −0.0 | +0.1 | +0.1 | −0.0 |
| mimetic | −7.8 | −6.4 | −5.1 | −2.8 | −1.7 |

Step at which test accuracy first reaches 75%, for seeds 0 / 1 / 2:

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| none | 1,800 | 1,700 | 1,900 |
| agent | 2,300 | 1,900 | 1,900 |
| random0 | 2,000 | 2,000 | 1,900 |
| mimetic | 2,400 | 2,300 | 2,500 |

## Locality over training

Each cell gives the mean attention distance in blocks 0–3 (averaged over heads; uniform = 4.14), then the attention mass on the 3×3 neighbourhood. Ranges are over the three seeds.

| Arm | Step 200, after the edit | Step 300 | Step 1,000 | Step 4,000 |
|---|---|---|---|---|
| none | 4.01–4.11 / 0.12 | 3.92–3.99 / 0.13 | 3.84–3.92 / 0.14 | 3.64–3.71 / 0.17 |
| agent | 2.62–3.37 / 0.20–0.37 | 3.11–3.65 / 0.16–0.27 | 3.21–3.60 / 0.18–0.25 | 3.10–3.38 / 0.21–0.27 |
| random0 | 4.02–4.08 / 0.12 | 3.90–3.99 / 0.13 | 3.78–3.93 / 0.14 | 3.60–3.72 / 0.17 |
| mimetic | 2.86–2.92 / 0.38 | 3.22–3.31 / 0.29 | 3.05–3.15 / 0.30 | 2.61–2.71 / 0.40 |
| fixed locality bias (recipe check, from step 0) | | | | 2.24–2.51 / 0.35–0.45 |

## Pass bar (fixed before the arms ran)

| Condition | Result |
|---|---|
| 1. Lasting gain: mean Δ_agent ≥ +0.5 and positive in 3 of 3 seeds, on final and last-5 | **Fails.** Mean −0.10 (final) and −0.12 (last-5); 2 of 3 seeds positive |
| 2. The random arm doesn't meet condition 1 | Holds (−0.20), but moot |
| 3. Head start: reaches 75% at least 30% sooner in 2 of 3 seeds | **Fails.** The agent arm reaches 75% 0–500 steps *later* |
| 4. Beats the scripted fix: mean (agent − mimetic) ≥ +0.3, positive in 2 of 3 seeds | Met (+1.42; −0.59, +2.31, +2.54), but only because the mimetic edit hurts |
| Barrier test: does mimetic meet condition 1? | **Fails.** −1.52, negative in every seed |

## Findings

- **No head start and no lasting gain.** Averaged over seeds, the agent arm is level with the control at every stage (−0.7 to +0.4 points by window) and at the end (−0.10).
  - The per-seed results range from −1.45 to +1.03. That is wider than the control's own seed spread (0.56 standard deviation) and wider than the random arm's range (−0.37 to −0.10).
  - So the agents' edits were not neutral, but their sign was not predictable.
- **The written locality partly survives, but buys nothing.**
  - The agent arms end with clearly more local early attention than the control: distance 3.1–3.4 against 3.6–3.7, and 3×3 mass 0.21–0.27 against 0.17.
  - Training kept part of the structure but didn't turn it into accuracy.
  - Mimetic keeps the most locality (2.6–2.7) and has the worst accuracy.
- **Mimetic applied mid-training is harmful.** Adding the identity structure to trained heads drops test accuracy to chance immediately (9–16%), and the run is still 1.7 points behind over steps 3,100–4,000. At step 0 this init is a known gain. At step 200 the rest of the network has already adapted to the old attention.
- **The agents' predictions were too optimistic.** All three predicted a final gain of +0.5 to +3 points:
  - Seed 2 came closest: +1.03 against +1.5 to +3.
  - Seed 0 got the sign wrong: −1.45 against +1 to +2.
  - My own prediction held for the mean: Δ between −0.3 and +0.5, with condition 1 failing. I did not predict that mimetic would hurt.
- **The headroom was small to begin with.** Locality hard-wired from step 0 was worth about +0.9 points in this 41-epoch recipe (one seed), so even a perfect edit could gain little. This recipe tests whether an edit can cross a small, real barrier. It isn't the data-starved regime of longer ViT training, where locality is worth several points.

## Cost

- **GPU:**
  - about 13 minutes for the recipe check (three concurrent runs)
  - 2.5 minutes for the base runs
  - 12 arm runs of 2–3 minutes each, one at a time
- **Wall-clock:** about one hour for the whole experiment, agent sessions included.
- **Agent sessions:** 20, 48 and 24 counted forward calls. The tool logs show 2–4 minutes between each session's first and last tool call.

## Not done

- **More seeds.** With per-seed agent effects from −1.45 to +1.03, three seeds can't separate a small mean effect from zero.
- **A longer, more data-starved recipe** (200+ epochs, or a CIFAR subset), where the locality gap is several points rather than 0.9.
- **Mimetic at step 0,** dropped for time.
- **A scripted Gaussian-bias edit at step 200,** matched to what the agents wrote. It would separate the agents' construction from the effect of writing locality at step 200 at all.

## Files

- **Code:** `vision/` (`model.py`, `data.py`, `data_hf.py`, `train.py`, `mimetic.py`, `agent_tools.py`, `agent_session.py`, `prompt_template.md`, `run_arms.sh`, `compare.py`)
- **Recipe check:** `runs/vision_vit/check/`
- **Base runs:** `runs/vision_vit/base/s{0,1,2}/`
- **Agent sessions** (prompt, notes, scripts, tool log, submission): `runs/vision_vit/s{0,1,2}_step200/`
- **Arms and control checkpoints:** `runs/vision_vit/arms/s{0,1,2}_step200/arm_{none,agent,random0,mimetic}/`
- **Plot:** `runs/vision_vit/compare.png`
