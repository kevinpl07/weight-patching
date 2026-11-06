<!-- Extracted from RESULTS.md (sections matching ['pilot']); full file in archive/reports/RESULTS.md -->

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


