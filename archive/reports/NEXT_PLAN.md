# Next experiments: where can weight editing beat the optimizer?

**Update (2026-11-04, late evening): two vision experiments have also been run and concluded,** a mid-training ViT locality edit and a training-free repair of spurious features in ResNet-50. See "Overall conclusion across all experiments" near the top of `FINDINGS.md` and the two vision follow-up sections there. In short, the ViT edit ended level with no edit. The spurious-feature edits beat naive training but lost to DFR-style balanced retraining, and a short script reproduces them.

Status (2026-11-04, updated in the evening): **#3 (pretraining phase transitions) has been run.** It gave a long head start that training erased; see "Outcome so far" at the end and the pretraining follow-up in `FINDINGS.md`. **A Sparks counterfactual has also been run:** plain GRPO fixed the thinking-mode loop in about 15 steps, which weakens the premise of #1 and #2 (see "Sparks counterfactual" at the end). The others are still candidates. Pick one, then write its full plan (arms, pass bar, predictions) before running anything.

## What we've learned so far

- **Toy pilots** ([FINDINGS.md](FINDINGS.md), first part): Opus edits to a grokking MLP gave large early gains that AdamW erased. Only the harmful part lasted: neurons the agent switched off never revived, because a dead ReLU gets no gradient.
- **RL head-start experiment** ([PLAN_2026-11-03_rl_head_start.md](PLAN_2026-11-03_rl_head_start.md), results in [RESULTS.md](RESULTS.md) and [FINDINGS.md](FINDINGS.md)):
  - The agent's anti-hedging edit gave Qwen3.5-0.8B a 7.2-point head start for GRPO, and training erased it within about 10 steps on both seeds.
  - The agent's *diagnosis* was right: the hedging direction is real and causal. But its edit was redundant, because GRPO's reward targets the same failure and fixes it in about 15 steps.
- **What makes editing work in Sparks:** a real failure with a visible symptom, and no optimizer running afterwards.

**The principle behind all of this:**

> An edit is worth something only if the optimizer would take a long time to make the same change, and it lasts only where the gradient doesn't push back.

So far we've only tested regimes where the optimizer is fast and has a strong signal. The candidates below invert that: the optimizer is stuck, slow or blind to the target, and the edit only has to *unlock* progress, not survive it.

**What carries over to any next experiment:**
- **Seeds:** two runs of the identical arm ended 4–8 points apart on every metric. Plan for 3+ seeds per arm, or pick setups cheap enough for 5–10.
- **Cheap baselines:** every candidate needs the cheap alternatives (prompting, a few SFT examples, rollback, sampling settings) as arms. An edit has to beat those, not only "no intervention".
- **Infrastructure we have:** `rl/grpo.py` (`--init`, `--eval-at`, `--trunc-zero`, temperature-scaled log-probs), `rl/eval.py`, `rl/run_arms.ps1`, `rl/evals_table.py`, `rl/ckpt_analysis.py`, and the agent tooling in `rl/agent_tools.py` and `runs/agent_dry/`.

## Candidates

### 1. Rescue a stuck RL run (recommended first)

**Hypothesis:** when RL is stuck with zero gradient, an agent's edit that restores even a small success rate restarts training, and does so faster than rollback or a few SFT examples. The edit doesn't need to last; it only needs to start the gradient flowing.

**Why it fits:**
- **It comes from our own failure.** The first arm A collapsed under the original reward: answer, `</think>`, answer again, never ending, until the 512-token limit.
- **It's a true zero-gradient trap.** Under the fixed reward (`--trunc-zero`), the collapsed model has every sample truncated and every reward 0. Every group then has zero advantage, so GRPO gets exactly zero gradient and can't recover by itself.
- **The agent's task suits its strengths:** a specific failure with a visible symptom, the same kind it diagnosed correctly before.

**Sketch:**
1. **Make the stuck checkpoint.** Resume `runs/grpo/armA_s0/ckpt/step_00075` with the old reward until about step 110, when truncation is about 100%, and save it. About 20 minutes; step ~120 was never saved.
2. **Arms, all continuing under `--trunc-zero`:**
   - (a) no intervention (expected: no learning at all)
   - (b) agent edit: weights plus rollouts, restore the end-of-answer behaviour
   - (c) random edit of the same size
   - (d) rollback to step 75
   - (e) SFT on 8–16 good answers
   - (f) a raised token limit
3. **Measure:** steps (and wall-clock, including agent time) until GSM-Symbolic sampled accuracy is back at the step-75 level (54.4%).

**Main risk:** rollback or a tiny SFT may be just as fast and much cheaper. That would still be a clear answer.

**Cost:** about 1 day; nearly all the infrastructure exists.

### 2. Fine-tuning problems with no signal (generalizes #1)

**Hypothesis:** in sparse-reward RL, where groups are all-wrong and the advantage is zero, an edit that unlocks a behaviour the model already has (output format, stopping, a required schema) lets RL start, and the edit's later fading doesn't matter.

**Why it fits:** "hard" has to mean *no signal*, not just *slow*. Grokking was slow, and the agent still lost because its edits fought the loss. Zero-advantage groups give exactly zero gradient, which is where an edit can't be "made redundant".

**Sketch:** a task where pass@64 is about 0 because of a behavioural gate, not missing ability. Examples: a strict output schema the model never follows, or a hard GSM subset where it never finishes in budget. Arms: edit, prompting, few-shot SFT, a higher sampling temperature, and no intervention.

**Main risk:** prompting or a few SFT examples may unlock the same signal more cheaply.

**Cost:** 1–2 days.

### 3. Pretraining phase transitions

**Hypothesis:** writing a known circuit, such as induction heads, into a small transformer before training skips the loss plateau that precedes its natural formation. That would save pretraining steps that last, unlike a generic head start.

**Why it fits:** early pretraining has a dense signal everywhere, so a generic head start is worth only a few steps. But phase transitions are well-documented *plateaus followed by sudden jumps*: the optimizer is slow, and the target structure is known and writable. Following the toy-pilot lesson, the edits *add* structure and never remove capacity.

**Sketch:** a 10–50M-parameter GPT on TinyStories (hours on the 4070, so 5–10 seeds per arm). Arms: written induction heads at init, a random edit of the same size, plain init, and a known-good init trick as a reference. Measure: steps to the end of the plateau, and tokens to reach the final loss.

**Main risk:** written circuits may not match what training builds, and training may tear them down and rebuild its own.

**Cost:** 2–3 days, mostly a new pretraining pipeline.

### 4. Repair parameters that training never reaches

**Hypothesis:** edits last exactly where the gradient is zero. Repairs to parameters that training doesn't update (under-trained token embeddings, dead neurons, never-firing features) will survive continued training almost completely.

**Why it fits:** it's the mirror image of the toy pilots, where harmful removals lasted because dead neurons get no gradient. Beneficial repairs in the same kind of place should last for the same reason. It's also a failure the training signal can't fix, which is one of Sparks' conditions.

**Sketch:** detect under-trained tokens in Qwen3.5-0.8B (behind the known "glitch token" failures), let the agent repair them, then continue fine-tuning on unrelated data. Check that the repair survives, and compare with fine-tuning on examples that contain those tokens.

**Main risk:** finding a failure of this kind that actually matters for any benchmark.

**Cost:** 1–2 days.

### 5. Route the agent's diagnosis through the optimizer

**Hypothesis:** the agent's diagnosis transfers but its written values don't. An agent that turns its diagnosis into *training interventions*, rather than weight values, beats both weight editing and no intervention. Examples: an auxiliary loss on a direction, a reward term, data selection, freezing, per-module learning rates.

**Why it fits:** it uses the agent's strength (reading and diagnosing) and avoids its weakness (predicting what the optimizer does over many steps).

**Test case:** the reward-hacking collapse.
- **Early signal:** in the collapsed run, the repeat behaviour was visible in 6 of 64 rollouts at step 70, well before the collapse at step 89.
- **The question:** could an agent that monitors activations and rollouts spot it forming and design a penalty that prevents the collapse?
- **Comparison:** the hand-written `--trunc-zero` fix.

**Main risk:** hard to make the comparison fair, since the agent effectively becomes a training engineer.

**Cost:** 2–3 days.

## Ranking

| Candidate | Fit with our lessons | Cost on our setup | Main risk |
|---|---|---|---|
| 1. Rescue a stuck RL run | Very high | About 1 day; infrastructure exists | Rollback or tiny SFT just as good |
| 2. Fine-tuning with no signal | High | 1–2 days | Prompting/SFT unlock it too |
| 4. Unreached parameters | High | 1–2 days | Hard to find a failure that matters |
| 3. Pretraining phase transitions | Medium-high | 2–3 days (new pipeline) | Written circuits don't match training's |
| 5. Diagnosis through the optimizer | Medium | 2–3 days | Hard to keep the comparison fair |

**Recommendation:** start with #1. It reuses everything we have and directly tests the core claim ("unlock, don't persist"), and its baselines are obvious. If editing beats rollback there, #2 generalizes it. #3 is the best longer-term direction for pretraining.

## Loose ends from the RL head-start experiment

These are optional and don't block any candidate:
- **Hedging-direction analysis** of the saved checkpoints (`rl/ckpt_analysis.py`): does GRPO move along the agent's direction by itself? About 30 minutes of GPU time.
- **The fixed arm D** (training at T=0.8 with temperature-scaled log-probs): does the edit beat simply sampling cooler? About 1.6 hours.
- **Disk:** the RL runs use about 75 GB under `runs/grpo/`. The step-15/30/50 checkpoints of finished runs can go if space is needed; keep step 100 and `armA_s0/ckpt/step_00075` (needed for candidate #1).

## Outcome so far (2026-11-04)

### Desk research on candidates 1–4 (four parallel research agents)
- **#1 (rescue a stuck RL run):** fit 5/10, our own collapse is the best case found.
  - Its trap is only *near*-zero gradient: 0 stops in about 1,900 samples still allows a stop rate up to 0.16%, so the run may recover by itself.
  - Rollback, a `</think>` stop string, or SFT on the model's own truncated outputs would probably tie an edit.
  - No good outside candidate (entropy collapse, zero-reward graph search, strict-format stalls) at ≤1.5B.
- **#2 (no-signal fine-tuning):** fit 6/10, but it is the same mechanism as #1: Qwen3.5-0.8B in thinking mode never closes `</think>`.
  - Constructed format gates are unlocked for free by prompting or by 8 SFT examples rewritten from the model's own outputs.
  - A 1-hour pre-check (official sampling settings, "be brief" prompts, 8-example SFT) would settle it.
- **#4 (unreached parameters):** fit 5/10.
  - The defect is real: Qwen3.5-0.8B has about 97 near-dead tied embedding rows, mostly Thai, Japanese and Chinese site boilerplate and code identifiers, and fails to copy or translate them.
  - But no benchmark touches them, and a 10-line script (mean of the sub-token embeddings) or a tokenizer ban fixes them just as well, so persistence alone can't separate an agent's edit from the script.
- **#3 (pretraining phase transitions):** prior work mostly answers the synthetic version (Singh et al. 2024, Gopalani & Hu 2025). Planting the full circuit is the solution, so it survives trivially. Only partial plants, or natural data, were open questions, so we ran it on natural data.

### #3 as run: induction plateau in small-LM pretraining
- **Test bed:** a 2-layer attention-only model (d512) on Python code. Its training loss shows a real plateau and a jump at about step 1,900.
- **Single Opus edit at step 1,000:** a hand-built previous-token and induction circuit. It led by up to 0.33 nats and about 650 steps, the gap closed once the control formed induction, and the final loss gap was 0.006, which is noise.
- **Two later sessions at step 3,000,** one with a look-ahead training tool and the final loss as the objective, both submitted empty edits.

**Lesson:** a plateau makes the window in which an edit matters longer (about 1,500 steps against about 10 in RL), but not permanent. Once the structure exists, everything left to learn is statistics, which AdamW learns better than a hand edit.

### Sparks counterfactual: plain RL on the thinking loop
- **Run:** 30 GRPO steps from base Qwen3.5-0.8B in thinking mode (4k tokens, T = 0.6 without top-k or top-p, zero reward for truncated answers), with no edit.
- **Result:** on 128 held-out problems with Sparks' sampling, accuracy went from 3.9% to 63.3% and loops from 61% to 3.9%. Most of the fix came in the first 15 steps.
- **For #1 and #2:** the thinking-mode loop isn't a zero-gradient trap. About 5% of answers finish at the start, and that's enough signal. A real trap needs a success rate that is truly zero, such as `armA_s0` after its collapse.
- **Open:** side effects (IFEval and non-thinking math on `runs/grpo/thinkloop_s0/ckpt/step_00030`), more seeds, and Sparks' exact sampling during training.

### Still open
- **Barriers rather than plateaus:** changes the loss would reward but the optimizer can't reach in budget. For example, multiplication-style local optima (Bai et al. 2025), or structure the training loss can't tell apart, such as length generalization.
- **Deeper models with MLPs** (the 6L GPT, `runs/pretrain_code/gpt6L_d384_s0`): more circuits that may form late or badly.
