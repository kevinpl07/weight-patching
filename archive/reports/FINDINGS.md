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

## Follow-up: an agent's edit at the start of LLM RL training (2026-11-04)

**Short answer: also no.** We tested the closest thing to the Sparks setup that still involves training: an edit that fixes a real, observed failure in a pretrained LLM, followed by RL training. The edit worked as an edit, but GRPO made it irrelevant within about 10 steps. Plan: [PLAN_2026-11-03_rl_head_start.md](PLAN_2026-11-03_rl_head_start.md); full tables: `RESULTS.md`.

### Setup
- **Model:** Qwen3.5-0.8B in non-thinking mode. Its main failure is rambling: at T=1, 57% of GSM-Symbolic answers hit the 512-token limit.
- **The edit:** in an earlier dry run, Opus 5.5 found a "hedging" direction in the residual stream and removed it, changing 4,089 weights in 4 MLP columns. On held-out sets, the edit raised sampled accuracy by 11–13 points, and a mechanism check showed it acts much like sampling at T=0.8 plus an extra push against hedge words.
- **Training:** full-weight GRPO on GSM8K for 100 steps, lr 1e-6, no weight decay. This is gentle on edits, unlike the toy pilots' weight decay of 1.0.
- **Arms:** base (A), the agent's edit (B) and a same-size random edit (C), on 2 seeds for A and B. A fourth arm, D (training at T=0.8), was invalid because of a bug.

### What we found
1. **The edit gives a real but tiny head start.** At step 0, B leads A by 7.2 points on sampled GSM-Symbolic, while the random edit is level with A. But GRPO alone takes A from 27% to about 43% in 15 steps, so the lead is worth about 7 steps of training.
2. **Training erases it, and there is nothing to erase in a healthy way.** Averaged over two seeds, B − A was −0.1 at step 15, −1.4 at step 30, −3.4 at step 50 and −0.3 at step 100. From step 15 on, B is indistinguishable from the random-edit control: both end +1.2 above A on seed 0.
3. **What lasts is the edit's cost, weakly.** The edit lowers greedy accuracy by 3.8 points at step 0, and B is still 3.7 below A on greedy at step 100 (both seeds negative). The two A seeds end 7.6 points apart on greedy, so this is a weak signal. It does echo pilot 1, where only the harmful part of an edit survived training.
4. **The optimizer already does what the agent did.** The rambling the agent's edit addressed is exactly what GRPO's reward punishes, and GRPO fixes it quickly: truncation goes from 57% to about 3% in 15 steps. Unlike the toy task, nothing here is undone by weight decay; the edit is simply made redundant.
5. **Seed noise dominates small effects.** Two runs of the identical arm ended 4–8 points apart on every metric. Single-seed differences of that size appeared and vanished several times during the night.

### Caveats
- **Only 2 seeds** for A and B, and 1 for C. The pass bar was amended after seeing arm A's first 75 steps, but before any other arm ran.
- **Untested:** whether the edit beats simply training at a lower temperature (arm D), and whether RL moves along the agent's hedging direction by itself (the checkpoints are saved; `rl/ckpt_analysis.py` is written).
- **Two training bugs** surfaced and were fixed. The reward ignored truncation, which caused an answer-repeat collapse in the first run. And arm D's log-probs weren't temperature-scaled. Neither affects the A/B/C results above.

### What this adds
Both conditions from "Why this differs from Sparks" were tested here:
- **A real failure with a visible symptom:** this held. The agent's diagnosis was correct, and its edit fixed the symptom.
- **An optimizer that won't fix or undo it:** this failed. RL training targets the same failure and fixes it faster than the head start is worth.

So editing before training is no better than editing during it, when the training objective already covers the failure. Weight edits look most useful where no further training follows, or for failures that the training signal doesn't reach.

## Follow-up: agent edits during small-LM pretraining (2026-11-04)

**Short answer: a long, real head start, and then the same story.** We looked for a regime where the optimizer is genuinely stuck rather than fast: the loss plateau before induction heads form in a small transformer trained from scratch. There, an Opus 5.5 edit gave by far the biggest and longest head start in this project: up to 0.33 nats and about 650 steps for roughly 1,500 steps. But AdamW closed the gap within about 1,000 steps of forming its own circuit, and the final loss differs by a noise-level 0.006. Two later sessions at step 3,000, the second with a look-ahead training tool and the final loss as the objective, found nothing worth editing and submitted empty edits. Full tables are in `RESULTS.md`.

### Setup
- **Pipeline (`pretrain/`):** small GPTs trained from random initialization on one RTX 4070. The run-length settings were 512-token context, 64 sequences per step, AdamW at lr 1e-3 with cosine decay, and 15,000 steps (about 490M tokens). Every 25 steps we probed loss, copy benefit and per-head previous-token and induction scores. Copy benefit is the loss on a passage minus the loss on an exact repeat of it.
- **Where we looked for a plateau (seed 0 each):**
  - TinyStories: no induction heads formed, and no plateau.
  - FineWeb-Edu and Python code: 4 model sizes, 2.7M–17M parameters.
- **The test bed:** Python code with a 2-layer attention-only model at width 512 (10.8M parameters). It is the only run where the plateau and the jump are visible in the training loss itself: about 900 flat steps, then a 0.34-nat drop between steps 1,875 and 2,450.
- **Arms:** each continued from the step-1,000 checkpoint with the same optimizer state and the same batches.
  - no edit: the control, which reproduces the original run to within 0.0005
  - Opus edit
  - random edit with the same entries and the same change norm per tensor

### What we found
1. **The agent diagnosed and built the missing circuit without being told.**
   - It wrote sinusoidal position features into the quietest dimensions, turned L0H6 into a previous-token head, and turned L1H2 into an induction head. That changed 295k entries, from statistics only.
   - Loss was 0.15 worse immediately, but training sharpened the circuit within 250 steps.
2. **A large, long head start.**

   | Val loss | Step 1,250 | 1,500 | 2,000 | 3,000 | 5,000 | 15,000 | Mean, 1k–15k |
   |---|---|---|---|---|---|---|---|
   | Agent edit | **2.749** | **2.619** | **2.464** | 2.320 | 2.153 | 1.905 | **2.104** |
   | No edit | 3.000 | 2.939 | 2.669 | 2.351 | 2.160 | 1.911 | 2.136 |
   | Random edit | 3.064 | 2.963 | 2.739 | 2.360 | 2.170 | 1.919 | 2.153 |

   - The edit saved about 650 steps at its best, against about 10 in the RL follow-up, because the optimizer really was stuck on a saddle. Induction needs two heads that only pay off together, so each one's gradient is a product of two small terms.
3. **Training still erases it.** The control forms its own induction circuit around steps 1,900–2,500, and the gap shrinks from 0.33 to 0.03 by step 3,000 and to about 0.007 by step 5,000. The final gap of 0.006 is the same size as the random edit's 0.008 in the other direction, so it is noise at one seed.
4. **Once the structure exists, there is nothing left to edit.**
   - At step 3,000, Opus found the planted circuit intact and tuned. It bounded the remaining copy-related gain at about 0.015 nats, below the roughly 0.055 cost of repurposing any head, and submitted an empty edit.
   - A second session at the same checkpoint was scored on the loss at step 15,000 and had paired 1,500-step look-ahead training runs. It investigated copy fidelity, layer 1's narrow token code, head redundancy, indentation tracking and Adam step sizes, and also submitted an empty edit.
   - **Its one structural candidate trailed persistently.** Rebuilding L0H5 as a second token-code head left the run about 0.011 behind for the last 800 look-ahead steps, with no crossover. A function-preserving QK rescale was noise.
5. **Damage persists just like gains fade.** A head training has built is worth roughly its removal cost for the rest of the run. Freeing it doesn't lead training to a better use of it.
6. **The capacity is used, not wasted.** In the trained control, every head's removal costs 0.09–1.76 nats. Pairs of heads are slightly complementary: joint removal costs a median 1.08× the sum of the single costs. The model is limited by having only 16 heads and no MLPs, not by idle heads. A 7.5M-parameter 4-layer GPT beats this 10.8M attention-only model, at 1.773 against 1.911.

### Caveats
- **One seed for everything.** No hand-written-circuit arm, so we don't know whether Opus's construction beats a textbook one.
- **The look-ahead tool loosened the original rules,** for the second step-3,000 session only. The agent used 3 of its 10 runs and about 21 of the roughly 120 minutes it was given.
- **The token embedding and unembedding were out of scope,** and most of the remaining knowledge lives there.

### What this adds
- **The principle holds even where the optimizer is genuinely slow.** An edit is worth the steps the optimizer needs to make the same change, and in a healthy run it eventually makes it. A plateau makes that window longer: about 1,500 steps here, against about 10 in RL. It doesn't make it permanent.
- **Where a lasting gain could still exist:** cases where the loss would reward a change but the optimizer can't reach it within the budget, such as a true barrier rather than a plateau, or structure the training loss can't distinguish.
- **The economics:** the edit took 18 minutes and 173k tokens to save about 600 steps, which is about 40 seconds of GPU time.

## Follow-up: would plain RL have fixed Sparks' reasoning loop? (2026-11-04)

**Short answer: yes, in about 15 steps.** Every experiment above pointed to the same pattern: when the training objective covers a failure, training fixes it. So we tested the counterfactual Sparks never ran. Sparks' Experiment 8 fixed thinking-mode reasoning loops in Qwen3.5-0.8B with a weight edit and no training afterwards. Here, 30 steps of standard GRPO fixed the same failure, and raised accuracy far beyond the loop fix. Full tables are in `RESULTS.md`.

### Setup
- **Model:** base Qwen3.5-0.8B in thinking mode, with a 4,096-token limit. It reproduces Sparks' failure: 99% of answers hit the limit and 61% are verbatim loops.
- **Training:** full-weight GRPO on GSM8K, with the same settings as the earlier RL arms. Only text after `</think>` is scored, and answers that run to the limit get zero reward. 30 steps, one seed, sampling at T = 0.6.
- **Held-out test:** the same 128 GSM8K test problems as the base measurement, with Sparks' sampling settings.
- **No edit arm.** The question was only whether training alone fixes the failure.

### What we found

| Held-out test | Accuracy | Loops | Hit the limit |
|---|---|---|---|
| Base model | 3.9% | 61% | 99% |
| After 30 GRPO steps | **63.3%** | **3.9%** | **3.9%** |

1. **Training escapes the loop quickly.** The share of training answers that finish thinking went from 4% to 30% by steps 3–5, 77% by steps 11–15, and 94% by the end. Loops fell from 39% to about 1%.
2. **It isn't a zero-gradient trap.** We predicted a stall. But about 5% of answers finished at the start, which left 2–3 of 8 groups per step with a learning signal, and that was enough.
3. **Training does more than an edit.** An edit removes a symptom. RL removed it and lifted held-out accuracy from 4% to 63%.
4. **The cost is small:** about 1.6 hours on an RTX 4070 to get loops mostly gone (step 13), and 3.7 hours for all 30 steps.

### Caveats
- **One seed,** although the effect is far beyond the 4–8 points of noise between seeds seen earlier.
- **Side effects weren't measured.** RL changes every weight, while Sparks' edit changed 0.016% of them. If RL damaged instruction following or non-thinking answers, the edit keeps a real advantage. This is the main open question.
- **A narrower setting than Sparks':** a 4k-token limit instead of 32k, and training sampled without top-k and top-p. That makes finishing more common at the start (about 5%, against 1 in 128 under Sparks' sampling), so the escape may be harder under their exact settings.

### What this adds
- **The answer to "would more training have fixed it" is yes, for Sparks' main LLM case as well.** The edit fixed a failure that a short, standard RL run fixes too, and the RL fix comes with a large accuracy gain.
- **What's left for editing:** cases where training isn't an option, where training would cause collateral damage an edit avoids, or where the success rate is truly zero, so there's no gradient at all.

## Follow-up: training-free repair of spurious features in an ImageNet classifier (2026-11-04)

**Short answer: the edits work, but they don't win, and they're a recipe a script can apply.** This is the setting the conclusion above asked for: a real failure with a visible symptom, and no training after the edit. Three blind Opus 5.5 sessions each repaired most of a pretrained ResNet-50's reliance on spurious cues by changing 3k–30k fc weights:
- The held-out spurious hit rate fell from 50% to 16–22%.
- Overall clean accuracy fell by 0.1–0.6 points.

None of the sessions passed the bar fixed in advance; they missed by a few photos.

**Two later arms settle the comparison:**
- **Scripted recipe:** a 30-line closed-form version of the recipe all three agents used (cut cue channels, boost object channels in each class's fc row) reproduces sessions 1 and 2 (session 2 exactly, session 1 to within 0.1 point of hit rate).
- **DFR-style retraining** (Kirichenko et al. 2022) of the last layer:
  - **Data:** about 40 labelled ImageNet photos per class, with each class's real photos balanced against its cue images, and an L2 pull toward the original weights.
  - **Selection:** chosen on dev data only.
  - **Result:** it is the only arm that passes the bar, narrowly (AP 0.583 against a 0.576 bar). It leaves somewhat more spurious hits than the best edits (21.5% against 15.7–17.2%) but does no damage to the repaired classes. It needs 7 s of GPU.
- **Naive training** (the earlier A1/A2 arms) loses to the edits, even at the same operating point and with the same ImageNet data.

**Naive training, stopped at the agents' operating point** (chosen without eval data):
- **Variants:** nine in total, including gentle KL-regularized ones and ones restricted to the same 50 fc rows.
- **Data:** first the agents' own dev data, then that plus about 39k labelled ImageNet-train photos.
- **Speed:** it reaches the same hit rate in 11–94 steps, with a few seconds of GPU training.
- **Spurious AP:** lower, 0.47–0.56 against the agents' 0.57–0.67.
- **Repaired classes:** more damage, 46–55% dev-class top-1 against 57–60%.
- **Overall accuracy:** only gentle fc training matches the agents here.

**A realistic stopping rule** (pool-val damage limits plus cross-validated cue hit) stops fc training with almost nothing fixed. It doesn't train the layer4 models at all.

By the pass bar fixed in advance, none of the 3 sessions passes:
- s2 missed the overall-accuracy limit by 5 of 7,000 photos.
- s3 missed the dev-class limit by 1–2 of 350 photos.
- s1 fell short on both the fix and the dev-class limit.

No naive-training checkpoint passes either, even one picked with held-out data; only DFR does. Plan, with post-hoc amendments A1 (fair stopping), A2 (ImageNet-train data) and A3 (DFR and the scripted recipe): [PLAN_2026-11-04_spurious_repair.md](PLAN_2026-11-04_spurious_repair.md); full tables: `RESULTS.md`.

### Setup
- **Model:** torchvision ResNet-50 (`IMAGENET1K_V1`), frozen. The edit is final.
- **Failure:** the Spurious ImageNet benchmark (Neuhaus et al., ICCV 2023). Its photos show only a class's cue, such as flowers for "bee" or underwater scenes for "snorkel", and are still classified as that class.
- **Data:**
  - **Dev,** seen by the agents and all baselines: 25 cue-only photos for each of 50 classes, plus 3 ImageNetV2 photos per class.
  - **Eval,** held out: 50 new cue-only photos per dev class, 7 ImageNetV2 photos per class, and 50 more benchmark classes the agents never saw.
- **Agent:** up to 200k entries (0.78%) of any tensor, forward passes and activations, no gradients or fitting, about 30 minutes.
- **Pass bar:** the hit rate halves and AP rises by 0.10. Overall top-1 drops by no more than 0.5 points and dev-class top-1 by no more than 3. The matched random edit fails the first condition.

### Results (held-out)

| Arm | Entries changed | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit |
|---|---|---|---|---|---|---|
| No fix | 0 | 50.2% | 0.476 | 62.91% | 62.3% | 45.1% |
| **Opus s1 / s2 / s3** | 3,169 / 5,000 / 29,912 | 21.8 / **15.7** / 17.2% | 0.570 / **0.670** / 0.594 | −0.13 / −0.57 / −0.14 | 57.1 / 59.7 / 58.9% | 47.1 / 46.9 / 47.5% |
| Random edits, same entries | same | 48.3–48.5% | 0.473–0.477 | −0.11 to +0.03 | 60.6–62.9% | 43.6–45.4% |
| Scripted projection (fc) | 102,450 | 0.2% | 0.772 | −1.43 | **24.6%** | 48.1% |
| SpuFix-style clamp | wrapper | 0.1% | 0.644 | −1.93 | 13.7% | 48.1% |
| Last-layer retraining, trained to ≤2% dev hit | 2.05M | 5.4% | 0.498 | −1.89 | 38.6% | 43.7% |
| Fine-tune layer4 + fc, trained to ≤2% dev hit | 17.0M | 4.6% | 0.619 | −0.99 | 50.3% | **35.9%** |

The fair comparison stops training where each agent stopped: the first step whose in-sample dev hit rate falls to that agent's (0.234 / 0.165 / 0.173). Ranges below are over the three matched points:

| Training on the agents' dev data | Steps (GPU s) | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit |
|---|---|---|---|---|---|---|
| **Opus s1 / s2 / s3** (for reference) | about 4–5 min sessions | 21.8 / 15.7 / 17.2% | **0.570 / 0.670 / 0.594** | −0.13 / −0.57 / −0.14 | **57.1 / 59.7 / 58.9%** | 47.1 / 46.9 / 47.5% |
| fc | 11–15 (0.1) | 17.3–23.5% | 0.476–0.478 | −0.60 to −0.81 | 48.0–50.0% | 44.9–45.5% |
| fc, 50 dev rows only | 21–29 (0.15) | 17.2–23.4% | 0.471–0.476 | −0.29 to −0.37 | 46.3–49.1% | 47.7–48.0% |
| fc + KL to the original, lr 2e-5 | 65–94 (0.6) | 18.3–23.9% | 0.481–0.485 | −0.04 to −0.31 | 48.6–51.1% | 46.3–46.4% |
| layer4 + fc | 28–42 (1.3) | 20.3–23.4% | 0.544–0.555 | −0.70 to −0.90 | 52.6–53.1% | 41.1–41.2% |
| layer4 + fc + KL | 30–50 (1.6) | 20.4–24.5% | 0.546–0.562 | −0.23 to −0.41 | 54.3–55.1% | 43.0–44.0% |

**With realistic labelled data (amendment A2).** Positives are about 39k ImageNet-train photos (up to 40 per class) plus the dev photos; negatives are the same 25 cue images per class:

| Training with ImageNet-train data | Steps (GPU s) | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit |
|---|---|---|---|---|---|---|
| fc, 50 dev rows only, matched | 21–28 (0.7) | 17.3–23.1% | 0.467–0.468 | −0.26 to −0.31 | 47.1–49.1% | 47.7–48.0% |
| fc + KL, matched | 60–80 (3) | 18.0–22.4% | 0.477–0.479 | +0.01 to −0.07 | 48.6–49.1% | 47.1–47.3% |
| layer4 + fc, matched | 20–40 (4) | 13.6–23.5% | 0.521–0.543 | −1.34 to −1.67 | 50.0–51.4% | 36.6–39.0% |
| layer4 + fc + KL, matched | 20–40 (4) | 16.4–25.2% | 0.522–0.544 | −0.84 to −1.10 | 51.7–53.4% | 39.3–40.5% |
| fc variants, practitioner rule | 5–17 | 41.0–43.0% | 0.471–0.473 | +0.01 to +0.11 | 58.6–59.4% | 45.8–46.3% |
| layer4 variants, practitioner rule | 0 (no step qualifies) | 50.2% | 0.476 | 0 | 62.3% | 45.1% |

`runs/vision_spurious/tradeoff.png` and `tradeoff_imagenet.png` plot every training trajectory on held-out data against the edits. They are diagnostic only. At equal hit rate, the agents sit 2–5 points above the best naive-training curve on dev-class top-1, and about 10 above the fc curves. No naive-training checkpoint on any curve meets the pass bar.

**DFR and the scripted recipe (amendment A3)**, both on cached penultimate features:

| Arm | Size / training | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit | Passes? |
|---|---|---|---|---|---|---|---|
| Script, s1 / s2 settings, 3 dev photos | 3,169 / 5,000 entries | 21.8 / 15.7% | 0.570 / 0.670 | −0.13 / −0.57 | 57.1 / 59.7% | 47.1 / 46.9% | no (identical to the agents) |
| Script, s1 / s2 settings, pool means | 2,527 / 4,792 entries | 34.1 / 29.1% | 0.534 / 0.653 | +0.04 / −0.76 | 61.7 / **70.0%** | 46.5 / 45.0% | no |
| **DFR, λ = 0.1, practitioner rule** | fc, 300 steps (7 s) | 21.5% | 0.583 | −0.29 | **62.9%** | 45.2% | **yes** |
| DFR, matched to s1 / s3 | fc, 27–170 steps | 22.0–26.4% | 0.540–0.577 | −0.24 to −0.49 | 60.3–62.6% | 43.5–45.8% | s3 match only |

In `tradeoff_imagenet.png`, the DFR curves with λ ≤ 0.1 lie above the agents on dev-class top-1 at every hit rate down to about 13%.

### What we found
1. **All three agents found the same mechanism on their own.**
   - Each class's cue is a set of context channels in the 2,048-d penultimate features, which class k's fc row weights positively.
   - The agents checked these are context detectors: the clean photos that fire them most are sharks and coral for "snorkel", and envelopes and menus for "quill".
   - All three shrank those weights in each class's fc row and boosted the class's object channels to compensate.
2. **On the trade-off between fixing the failure and hurting accuracy, the edits beat every baseline.** Session 2 gets a better spurious score than the layer4 fine-tune, with about half its accuracy loss, a fifth of the damage to the dev classes (−2.6 against −12 points), and 3,400× fewer changed weights.
3. **Plain training on the same images doesn't do the agents' job, even when stopped where the agents stopped.**
   - **fc training** reaches the agents' hit rate within 11–94 steps, mostly by lowering the class everywhere. Spurious AP doesn't move (0.47–0.49), and the repaired classes lose 11–16 points of accuracy.
   - **layer4 training** does better (AP up to 0.56, dev classes 53–55%), but still trails all three edits on both.
   - **Gentle, KL-regularized training** matches the careful agents (s1, s3) on overall accuracy and beats s2. So the edits' advantage is in keeping real photos of the repaired classes classified correctly, not in protecting the other 950 classes.
   - **Choosing a stopping point by cross-validation on dev** stops training after 2–15 steps, with almost nothing fixed. With 3 clean photos per class, training can't find a point that fixes the failure and passes the damage check. The agents did, from the same photos.
   - **Even picking checkpoints with held-out data,** no training checkpoint meets the pass bar. The best dev-class accuracy at the agents' hit rates is 55.7%.
   - **More labelled data doesn't change this.** With about 39k ImageNet-train photos, every matched training checkpoint still has lower AP (at most 0.54) and lower dev-class accuracy (at most 53%) than every edit. Layer4 training even loses 0.8–1.7 points of overall accuracy at those points.
   - **A practitioner's validation rule refuses the trade.** For the layer4 models, no step keeps pool-val accuracy within the damage limits, so it picks "don't train". For fc, it stops at a 41–43% hit rate. Fixing these classes costs accuracy on real photos of them, and training has no cheaper way to pay.
   - **The scripted projection** gets the best spurious score only by wiping out the classes themselves.
4. **The agents predicted their own results well.** 10 of their 12 held-out predictions landed in or better than the stated range. Their 3-fold estimates of damage, built from only 3 clean photos per class, held up. The one real miss: session 2's object boost made its classes steal 0.5 points of accuracy from other classes, against a forecast of −0.3 to +0.2.
5. **No transfer to unseen spurious features.** The edits are per class, and they slightly raise the hit rate on the unseen classes (+2 points, about at the noise edge), probably because the edited classes stop competing for those images. Only layer4 training, which changes features rather than the readout, generalizes (45% → 36–41%).
6. **The agents' contribution was the recipe, not per-class judgment.** A closed-form script reproduces sessions 1 and 2 (session 2 exactly; session 1 to within 0.1 point). With ImageNet clean means instead of 3 photos, the same script keeps the repaired classes more accurate (up to 70% against 62% unfixed) but removes fewer spurious hits.
7. **The right training method beats the edits.** Two ingredients separate DFR-style retraining from the naive arms: balancing each class's real photos (cue-containing ones oversampled) against its cue images, and an L2 pull toward the original fc. With them, a dev-only rule picks a model that passes every condition. Without the L2 term, it fails on overall accuracy.

### Caveats
- **The pass bar was strict relative to the noise.** The dev-class accuracy uses 350 photos (SE about 2.6 points), and the two near-misses are 1–5 photos. As fixed, the result is 0 of 3.
- **The spurious score is simplified.** It uses 7 ImageNetV2 positives per class instead of 50 ImageNet-val images, so it isn't comparable with the paper's table.
- **No real SpuFix.** It needs ImageNet-train activations. The SpuFix-style clamp uses directions estimated from dev data instead.
- **The fair training comparisons are post-hoc.**
  - **What changed:** the original training arms trained until the dev hit rate was at most 2%, an extreme the agents never chose. The matched-point variants (A1) and the ImageNet-data variants (A2) were added after all arms had been scored.
  - **Selection:** their stopping rules use no eval data, and no learning rates or loss weights were tuned.
  - **Not tried:** other regularizers or methods, such as group-robust losses or data augmentation, might do better.
- **The ImageNet-train photos in A2 are the ones ResNet-50 was originally trained on.** That is what a practitioner would have, so it's the right counterfactual. But the pool-val split is near-training data for the model (87.6% top-1 against 62.9% on ImageNetV2). So the practitioner rule's damage check sees a near-training distribution.
- **The agents' dev set is tiny,** with 3 clean photos per class. A2 shows that naive training's shortfall is not caused by that. DFR, though, uses about 40 labelled photos per class that the agents never had.
- **DFR's pass is thin and was found post-hoc.** AP clears the bar by 0.007. The practitioner rule picked the last step of a 300-step cap, and the DFR design came after all other results were known.
- **DFR's oversampling borrows the agents' idea.** The cue score it uses to find cue-containing class photos is the agents' cue-channel definition.

### What this adds
- **This is the first setting in the project where an edit clearly beats random edits and naive training on held-out data.** That holds at the same operating point, even when training gets about 14× more labelled data. It fits the conclusion's prediction: a specific failure with a visible symptom, and no optimizer afterwards.
- **But it is not a win for editing over training.** A standard method for exactly this problem (DFR-style balanced last-layer retraining, with a pull toward the original weights) passes the bar where the edits don't. It does so with seconds of GPU and about 40 labelled photos per class. And the edits themselves reduce to a short script.
- **What the agents did add:** within minutes and from 3 clean photos per class, they found the right mechanism (context channels in the fc readout) and a recipe that is better than any naive training and close to DFR. It also identifies cue-containing photos for DFR's oversampling. The diagnosis is the reusable part, more than the edit.
- **The side-effect question:** at an equal hit rate, the edits keep the repaired classes 3–13 points more accurate than naive training, but about 3–4 points less accurate than DFR.
- **The edits are local.** They repair the reported classes and don't touch the unseen ones. Fixing spurious features at scale would need either a report per class or an edit to the features themselves, which no agent attempted.

### Files
- **Plan:** `PLAN_2026-11-04_spurious_repair.md`
- **Code:** `vision_spurious/` (`eval.py`, `agent_tools.py`, `session.py`, `baselines.py`, `summarize.py`, `prompt_template.md`)
- **Data:** `data_vision/`
- **Runs:**
  - sessions: `runs/vision_spurious/session_{1,2,3}/`
  - baseline weights and logs: `runs/vision_spurious/arms/`
  - per-arm evals: `runs/vision_spurious/eval/`
  - fair training comparison: `vision_spurious/train_curves.py`, `runs/vision_spurious/curves/`, `runs/vision_spurious/tradeoff.png`
  - ImageNet-data training: `vision_spurious/imagenet_pool.py`, `vision_spurious/train_full.py`, `runs/vision_spurious/curves_in/`, `runs/vision_spurious/tradeoff_imagenet.png`
  - DFR and the scripted recipe: `vision_spurious/a3_arms.py`, `runs/vision_spurious/curves_in/dfr_lam*.json`, `runs/vision_spurious/summary_a3.txt`

## Follow-up: an agent's mid-training edit to give a ViT locality (2026-11-04)

**Short answer: no. The structure lasted, but the accuracy gain didn't.** This was the vision counterpart of the induction-plateau run, aimed at the open question of *barriers rather than plateaus*: ViTs trained from scratch on CIFAR-10 hardly learn local attention, and locality is known to raise their final accuracy.

At 5% of training, three blind Opus 5.5 sessions each wrote a Gaussian locality prior into the early attention heads. Part of that locality survived to the end of training. But the edited runs ended level with unedited training on average (−0.10 points; per seed −1.45, +0.12 and +1.03), with no head start along the way. The scripted reference, mimetic attention applied mid-training, cost 1.5 points. Plan: [PLAN_2026-11-04_vit_locality.md](PLAN_2026-11-04_vit_locality.md); full tables: `RESULTS.md`.

### Setup
- **Model (`vision/`):** a ViT with 2.69M parameters (patch 4, width 192, 6 blocks, 3 heads, learned position embeddings).
- **Training:** CIFAR-10 with crop and flip, AdamW and cosine decay, 4,000 steps (41 epochs). About 2–3 minutes per run on the RTX 4070.
- **Edit point:** step 200. Every arm resumes from the same checkpoint with the same optimizer state and batches. 3 seeds per arm.
- **Arms:**
  - no edit
  - Opus edit (about 30 minutes, training images only, budget 900k entries)
  - random edit matched per tensor
  - mimetic attention (Trockman & Kolter 2023) added to every layer
- **Headroom, from a check run before the arms:** a fixed locality bias from step 0 beat plain training by only about +0.9 points (82.5% against 81.6%).

### What we found

| Test accuracy at step 4,000 | Seed 0 | Seed 1 | Seed 2 | Mean | Δ vs no edit |
|---|---|---|---|---|---|
| No edit | 81.20 | 82.27 | 81.44 | 81.64 | |
| **Opus edit** | 79.75 | 82.39 | 82.47 | 81.54 | −0.10 (−1.45, +0.12, +1.03) |
| Random edit, same size | 81.10 | 81.90 | 81.32 | 81.44 | −0.20 |
| Mimetic attention at step 200 | 80.34 | 80.08 | 79.93 | 80.12 | −1.52 |

1. **The agents agreed on the diagnosis and built the same fix independently.** All three found near-uniform attention in every head and a weak position embedding at step 200. Each wrote clean position features into quiet residual directions, then turned 3–4 of the 64 Q/K dimensions of the early heads into a Gaussian distance bias, leaving the rest of each head untouched. The edits changed 22k–74k entries, under 3% of the model. Each agent checked dose–response.
2. **The pass bar failed.** The conditions were a mean gain of at least +0.5 points with 3 of 3 seeds positive, and reaching 75% accuracy 30% sooner. The agent arm reached 75% 0–500 steps *later* than the control and was level with it at every stage. Its seeds spread more widely than the random edit's, so the edits had an effect, but its sign wasn't predictable.
3. **The written structure survives, but buys nothing.** At step 4,000 the agent arms' early layers are still clearly more local than the control's: a mean attention distance of 3.1–3.4 patches against 3.6–3.7. Training kept the locality and simply didn't convert it into accuracy.
4. **The textbook fix applied mid-training is harmful.** Mimetic init added at step 200 drops accuracy to chance immediately, because downstream layers were tuned to the old attention, and the runs never catch up. It also keeps the most locality to the end. In this setting, locality written in mid-training isn't what helps; locality present from the start (+0.9) is.
5. **The agents were overconfident.** All three predicted a gain of +0.5 to +3 points. One got the sign wrong (−1.45 against +1 to +2).

### Caveats
- **Small headroom.** In this 41-epoch recipe, locality is worth less than a point even when built in from step 0. A longer or more data-starved run, where the literature reports a gap of several points, is the stronger test of the barrier idea, and is untested here.
- **3 seeds** against per-seed agent effects of −1.45 to +1.03.
- **No matched scripted Gaussian edit,** so we can't separate the agents' particular construction from the effect of writing locality at step 200 at all.
- **The mimetic arm is our mid-training adaptation** of a step-0 init, not the published method.

### What this adds
- **A barrier the optimizer can't cross is not enough by itself.** This optimizer does leave locality on the table: the control never becomes local. An edit can supply the locality, and training keeps it. But once the network has adapted to global attention, writing locality in late doesn't recover the from-scratch locality gain. The value of an inductive bias seems to come from shaping training from the start, not from the structure existing at the end.
- **Consistent with every mid-training result in this project:** an agent can read the network correctly and write the structure it intends, but the edits don't improve where training ends up.

### Files
- **Plan:** `PLAN_2026-11-04_vit_locality.md`
- **Code:** `vision/` (see `vision/README.md`)
- **Runs:**
  - recipe check: `runs/vision_vit/check/`
  - base runs: `runs/vision_vit/base/`
  - agent sessions: `runs/vision_vit/s{0,1,2}_step200/`
  - arms and the plot: `runs/vision_vit/arms/`, `runs/vision_vit/compare.png`

## Files

- **Write-ups:** `RESULTS.md` (all tables, per-round details, rule audits)
- **Pilot 1:** `experiment.py`, `agent_prompt.md`, `kill_controls.py`
- **Pilots 2 and 3:** `loop.py` (`--v2` for pilot 3), `loop_prompt.md`, `loop_prompt_v2.md`
- **Capacity check:** `capacity_check.py`
- **Agent tools:** `edit_tools.py`
- **Raw data:** `runs/seedN/`. Every agent session sits there with its prompt, tool log, scripts and submission.
- **RL follow-up:**
  - Plan with amendments: `PLAN_2026-11-03_rl_head_start.md`
  - Setup: `rl/README.md`
  - Code: `rl/grpo.py`, `rl/run_arms.ps1`, `rl/evals_table.py`, `rl/ckpt_analysis.py`
  - Runs: `runs/grpo/arm*_100/`
  - The agent's edit: `runs/agent_dry/`
- **Pretraining follow-up:**
  - Setup and scripts: `pretrain/README.md`
  - Baseline runs: `runs/pretrain/` (TinyStories), `runs/pretrain_fw/` (web text), `runs/pretrain_code/` (Python)
  - Agent sessions and arms:
    - `runs/pretrain_agent/code_d512_s0_step1000/`: prompt, notes, scripts, `arm_none`, `arm_agent`, `arm_random0`, `compare.png`
    - `runs/pretrain_agent/code_d512_s0_multi/`: the two step-3,000 sessions
- **Sparks counterfactual (RL on the thinking loop):**
  - Code: `rl/grpo.py --thinking`
  - Training run: `runs/grpo/thinkloop_s0/`
  - Held-out test: `runs/eval/thinkloop_s0_step30/`, against `runs/eval/base/`
- **Vision: spurious-feature repair (no training after the edit):**
  - Plan: `PLAN_2026-11-04_spurious_repair.md`
  - Code: `vision_spurious/` (`summarize.py` regenerates the table and pass-bar check)
  - Sessions and arms: `runs/vision_spurious/session_{1,2,3}/`, `runs/vision_spurious/eval/`, `runs/vision_spurious/summary.txt`
  - Fair training comparison (amendment A1): `vision_spurious/train_curves.py`, `runs/vision_spurious/curves/`, `runs/vision_spurious/summary_curves.txt`, `runs/vision_spurious/tradeoff.png`
  - Training with ImageNet-train data (amendment A2): `vision_spurious/imagenet_pool.py`, `vision_spurious/train_full.py`, `runs/vision_spurious/curves_in/`, `runs/vision_spurious/summary_curves_in.txt`, `runs/vision_spurious/tradeoff_imagenet.png`
  - DFR and the scripted recipe (amendment A3): `vision_spurious/a3_arms.py`, `runs/vision_spurious/curves_in/dfr_lam*.json`, `runs/vision_spurious/summary_a3.txt`
- **Vision: ViT locality edit mid-training:**
  - Plan: `PLAN_2026-11-04_vit_locality.md`
  - Code: `vision/` (see `vision/README.md`)
  - Sessions: `runs/vision_vit/s{0,1,2}_step200/`; arms and plot: `runs/vision_vit/arms/`, `runs/vision_vit/compare.png`
