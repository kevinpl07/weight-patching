<!-- Extracted from FINDINGS.md (sections matching ['(preamble)']); full file in archive/reports/FINDINGS.md -->

# Findings: can an LLM speed up training by editing weights mid-run?

**Short answer: no, not on a healthy training run.** Across three pilots (24 Opus 5.5 sessions, 3 seeds each), direct weight edits gave large early gains, but no Opus-edited run ever reached the 95% held-out accuracy that plain training reaches by itself in 23k–31k steps. The optimizer erases what an edit adds and keeps what an edit removes. The detailed logs and per-round tables are in `RESULTS.md`; this file is the summary.

## Overall conclusion across all experiments (as of 2026-11-04)

| Experiment | Training after the edit? | Did the edit help? | What beat or matched it |
|---|---|---|---|
| Grokking MLP, 3 pilots | yes | early gains, erased; removals stuck | plain AdamW |
| LLM RL head start (Qwen3.5-0.8B) | yes | 7-point head start, gone in ~10 steps | plain GRPO |
| Small-LM pretraining, induction plateau | yes | up to 650 steps ahead, gone by the end | plain AdamW |
| Sparks' thinking loop, no edit arm | — | — | 30 GRPO steps: 4% → 63% accuracy |
| ViT locality on CIFAR-10 | yes | level with no edit (−0.1 points) | plain AdamW |
| Spurious features in ResNet-50 | no | yes: beats random edits and naive training at equal operating point | DFR-style balanced retraining with ~40 photos/class; a short script reproduces the edits |

1. **Opus reads networks well.** In every experiment it found the real mechanism (Fourier circuit, hedging direction, missing induction circuit, uniform attention, context channels) and mostly predicted the immediate effect of its edits.
2. **When training continues, edits are worth only the steps the optimizer needs to make the same change.** That ranged from about 10 steps (RL) to about 1,500 (a plateau). The optimizer erased gains, kept structure that didn't help, and kept damage it couldn't undo.
3. **When no training follows, edits can do real work,** but the right standard method with ordinary data still matched or beat them, and the edit itself reduced to a recipe a script can apply.
4. **The reusable value is the diagnosis, not the written weights.** The best use of an agent here is finding the mechanism and designing the fix, then running it as a script or training procedure.

## The question

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, November 2026; logs at [github.com/RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)) showed frontier LLMs can read and write network weights directly, without an optimizer. We asked a narrower question: **if an LLM edits the weights in the middle of training, and training then continues, does the network reach high held-out accuracy sooner?**

## Setup

- **Task:** (a + b) mod 97, with 40% of all pairs used for training.
- **Model:** a 2-layer ReLU MLP (194 → 128 → 97, 37,473 weights).
- **Training:** full-batch AdamW, lr 1e-3, weight decay 1.0. The network memorizes the training set by about step 1,300 and generalizes ("groks") much later.
- **Edit point:** 1,000 steps after training accuracy reaches 100%, while held-out is at about 0.5%.
- **The agent:** Opus 5.5, which reads the weights and writes values through `edit_tools.py`. It may change at most 2% of the entries (749) per edit, and may never use an optimizer or fitting.
- **Pass bar, set before running:** reach 95% held-out at least 30% sooner than unedited in 2 of 3 seeds, while random edits of the same size don't.
- **Controls:**
  - unedited training
  - random edits matched to each Opus edit in entry count and size
  - random neuron removal
  - a scripted revival of dead neurons
  - narrower networks trained from scratch

## The three pilots

| Pilot | What changed | Best held-out per seed (0 / 1 / 2) | Reached 95%? |
|---|---|---|---|
| 1 | One edit at the edit point | 84.3% / 78.5% / 82.3% | 0 of 3 |
| 2 | Four edits, 4k steps apart; each agent sees the history so far | 86.2% / 76.7% / 90.9% | 0 of 3 |
| 3 | Pilot 2 plus a reframed prompt, a read-only optimizer view, optimizer resets, 1,000-step trial runs, and predictions shown next to outcomes | 78.9% / 85.9% / 93.0% | 0 of 3 |
| — | Unedited (plain AdamW) | reaches 95% at 30.9k / 22.9k / 28.9k steps | 3 of 3 |
| — | Scripted revival of dead neurons at +4k | reaches 95% at 40.8k / 24.1k / 24.6k steps | 3 of 3 |
| — | Random-edit loops (pilots 2 and 3, 12 runs) | 30k–53k steps when they get there | 10 of 12 |

The pass bar failed in every pilot. The random-edit loops, which make edits of the same size in random places, did better than the Opus loops in the long run.

## What we found

### 1. The early gains are real
Most Opus edits improved held-out accuracy in the short term, by up to 24 points on the spot, and Opus runs reached 50% held-out 1.8–3.4× sooner than plain training. Averaged over the first 12k steps, the pilot 2 and 3 runs were 15–41 points ahead of plain training. Opus reads the weights well: it found the textbook Fourier circuit in every session and correctly predicted the immediate effect of its edits.

### 2. Training erases those gains
Pilot 2 shows the pattern most clearly: a jump right after the edit, then decay over the next 4,000 steps. Seed 0 round 3 went 84% → 77%. After the last edit, the runs froze: held-out moved by less than 0.5 points over 44,000 steps.

### 3. The one lasting effect is the harmful one: lost capacity
- In every pilot, agents "switched off memorizing neurons" by setting a large negative bias, cutting 54–86 of 128 neurons in a single edit.
- A ReLU that never fires gets no gradient, so those neurons never came back.
- Trained from scratch, networks with 54–84 neurons never reach 95%; they top out at 76–92%. 128 neurons is already close to the minimum this task needs.
- The Opus runs mostly ended where networks of their remaining width end.

### 4. Better tools changed behaviour, but not the outcome
Pilot 3's tools made the agents visibly better:
- **Predictions became accurate.** Round 1 forecasts were off by up to 31 points. Once agents saw earlier forecasts next to outcomes, all 9 later forecasts landed within 4 points.
- **Agents caught bad ideas.** Trial runs showed that reviving all dead neurons makes them memorize, and the agents dropped it.
- **Agents rebuilt capacity on purpose,** as noise-cancelling partner neurons. Seed 2 climbed steadily to 93%, the best Opus run, and the only one ahead of plain training on average over the whole run.

But the round-1 cuts happened anyway, and the 1,000-step trials actively encouraged them: a cut looks great after 1,000 steps, and its cost shows only after 10,000 or more. Rebuilding neurons costs 100–300 budget entries each, too slow to recover the lost capacity.

## Why: how AdamW works against edits

1. **The gradient rebuilds whatever lowers training loss.** Edits that remove memorization raise training loss, and the quickest way back down is to memorize again.
2. **Adam moves every weight at full speed.** Each step moves each weight about lr = 0.001, regardless of gradient size. Over 4,000 steps that's up to 4 units, against weights of about ±1, so any written value can be fully rewritten in one round.
3. **Weight decay erases anything the gradient doesn't support.** A weight with no gradient behind it loses about two-thirds of its size in 1,000 steps. Hand-written offsets and new partner neurons fade unless training loss pulls them up.
4. **The asymmetry.** Adam can undo additions but not removals: a dead ReLU stays dead. So every edit is filtered so that its harm stays and its benefit fades.

An edit lasts only if it moves the network somewhere AdamW also wants to go. Edits like that are ones AdamW would eventually find itself, so the most an agent can offer on a healthy run is a head start.

## Why this differs from Sparks

| | Sparks | This project |
|---|---|---|
| What's scored | The edited network, as is | Where training goes after the edit |
| What the agent does | Debugs a specific failure with a visible symptom (a reasoning loop, a policy stuck in a local minimum) | Generic optimization by theory (cleaner frequencies, fewer memorizing neurons) |
| What happens after the edit | Nothing: the edit is final | AdamW keeps running and pursues its own objective |

Weight editing pays off when two things hold:
1. **There is a specific failure with a visible symptom**, so the agent debugs instead of optimizing blind.
2. **The optimizer won't fix or undo it**, either because training is stuck or because it's over.

Mid-training editing of a healthy run has neither. Sparks' own negative result points the same way: Gemma, a model with no single bug, couldn't be improved by editing.

## Caveats

- **One tiny task with a famous solution.** Opus may be recognizing the textbook circuit rather than reading this network's weights; we never ran the planned check with relabeled inputs. The very strong weight decay (1.0) makes this setup unusually hard on edits.
- **Small samples.** 3 seeds per arm and a 60k-step horizon. "Never reached 95%" means within 60k steps.
- **Pilot 3's trials show held-out accuracy,** so agents chose between up to 3 candidates per round partly on test data.
- **The economics.** Even a perfect win here would not have been useful: a full run takes about 35 seconds on an RTX 4070, while each agent round takes 4–11 minutes and about 120k tokens.

## What would be worth testing next

A real, unplanted failure in a trained model, with a symptom you can observe, fixed by editing, with no retraining afterwards. That is the Sparks setup, and it plays to what Opus showed it does well here: reading weights, forming a correct diagnosis, and predicting the immediate effect of an edit. Speeding up a healthy training run looks like a dead end.

