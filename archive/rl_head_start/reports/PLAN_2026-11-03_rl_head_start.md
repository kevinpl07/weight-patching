# Plan (concluded 2026-11-04): does an agent's weight edit help an RL training run?

> **Outcome (2026-11-04): no.** Starting GRPO from the agent's edit gave a 7.2-point head start on GSM-Symbolic sampled accuracy, and training erased it within about 10 steps on both seeds. Averaged over seeds 0 and 1, B − A was −0.1 at step 15 and −0.3 at step 100, about the same as the random-edit control. The pass bar failed (conditions 1, 2 and 3). Arm D was invalid because of a temperature bug and was not rerun. Full tables are in [RESULTS.md](RESULTS.md) and the summary is in [FINDINGS.md](FINDINGS.md). Everything below is the plan as written, plus the amendments made during the run.

## Background

- **Inspiration:** [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang and Khattab, MIT CSAIL, November 2026; logs at [github.com/RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)). Their Experiment 8 fixed reasoning loops in Qwen3.5-0.8B by editing 0.016% of its weights.
- **Our toy pilots** ([FINDINGS.md](FINDINGS.md), [RESULTS.md](RESULTS.md)): mid-training edits on a healthy run gave early gains that AdamW then erased (lr 1e-3, weight decay 1.0).
- **Setup and baselines:** [rl/README.md](rl/README.md). The model is Qwen3.5-0.8B in non-thinking mode, with a 512-token answer limit, trained by full-weight GRPO on GSM8K natively on Windows (about 36 s per step).

## What we know so far

1. **The base model's main failure is rambling, not looping.** Under sampling at T=1, 49% (GSM8K) to 60% (GSM-Symbolic) of answers hit the 512-token limit and score nothing. Finished answers are about 60% correct, and 18–34% of the cut-off answers had already written the right number.
2. **The agent dry run** (`runs/agent_dry/`): Opus with 20 forward passes and the GSM8K training set only.
   - Its first edit, which suppressed detour words like "Wait" and "Alternatively", failed: the model just wandered in other words. The agent dropped it.
   - It then found a consistent "hedging mode" direction and removed it with a 4,089-weight edit in 4 `down_proj` columns (layers 15, 16, 19, 21).
3. **Independent scoring** of that edit on sets the agent never saw:

   | | Base | Edit | Random edit, same size | Base at T=0.8 |
   |---|---|---|---|---|
   | GSM-Symbolic sampled (T=1 unless noted) | 24.0% | **37.2%** (+13.2, z=6.1) | 27.2% (n.s.) | **36.6%** |
   | GSM8K sampled | 30.9% | **42.1%** (+11.2, z=7.7) | – | – |
   | GSM-Symbolic greedy | 45.8% | 42.2% (−3.6, n.s.) | 48.0% (n.s.) | – |
   | IFEval prompt-level strict | 52.9% | 55.5% (n.s.) | 54.5% | – |

4. **Mechanism check** (`rl/mech_check.py`, 200 base answers, about 92k tokens):
   - The edit lowers next-token entropy everywhere by almost exactly what T=0.8 does.
   - On top of that, it cuts the probability of hedge words about 6× more than T=0.8 does.
   - So the edit is roughly **"temperature 0.8 plus a specific anti-hedging push"**. Its accuracy gain matches simply sampling at T=0.8.

## The question

GRPO samples at temperature 1, so the failure the edit fixes is exactly what training sees. **Does starting GRPO from the edited model make training go faster or end better? Does training keep the fix or wash it out? And is any of that more than just training at a lower temperature?**

## Design: four arms from step 0

All arms use the same settings: `rl/grpo.py`, 8 problems × 8 samples per step, 512 tokens, lr 1e-6, no weight decay, 8-bit AdamW, the same data order, 300 steps.

| Arm | Starts from | Sampling temperature in training | What it tells us |
|---|---|---|---|
| **A** | Base | 1.0 | Reference: how fast GRPO fixes rambling by itself |
| **B** | Base + agent edit (`runs/agent_dry/model_edit`) | 1.0 | Head start? Does it last? |
| **C** | Base + matched random edit (`runs/agent_dry/model_random`) | 1.0 | Rules out "any perturbation helps" |
| **D** | Base | 0.8 | The cheap alternative: just sample less randomly in training |

### Measurements

- **Quick eval every 75 steps (0, 75, 150, 225, 300),** about 10 minutes each:
  - GSM-Symbolic (500), greedy and **sampled at T=1**. Every arm is scored at T=1 and greedy, including D.
  - IFEval (first 200 prompts), greedy.
  - Truncation, length and language mixing for each.
- **Training logs every step:** reward, accuracy, truncation, length.
- **Does training learn what the agent found?** For each checkpoint, offline from the saved checkpoints:
  - Teacher-force a fixed set of 64 base answers.
  - Measure (a) the projection on the agent's hedging direction (`runs/agent_dry/work/seqlevel.pt`) at layers 15, 16, 19, 21, and (b) `mech_check.py`: entropy and hedge-word log-probability against the base model.
  - If arm A drives the same direction down on its own, the agent predicted what RL learns, and B is a head start on the optimizer's path.
- **Full eval on the final checkpoints:** `rl/eval.py` (GSM8K test, GSM-Symbolic, IFEval).

### Pass bar (fixed before running)

- **Primary metric:** GSM-Symbolic sampled accuracy (T=1).
- **B "helps training"** if all of these hold:
  1. B reaches A's step-300 accuracy by step 210 or earlier (at least 30% sooner), **or** B ends at least 3 points above A at step 300.
  2. At step 300, B is no more than 2 points below A on GSM-Symbolic greedy and on IFEval.
  3. C does not meet condition 1.
- **B "beats the cheap alternative"** only if it also meets condition 1 against D.
- **Seeds:** single-seed results count as a signal, not a result. A result needs seed 1 for at least A and B. GPU kernels make runs drift apart after step 1, so seeds are the only noise control.

### Predictions (written before running)

- B starts about 13 points ahead of A on sampled accuracy.
- A catches up within roughly 100–150 steps, because GRPO quickly learns to shorten and commit.
- D tracks B closely early on.
- Most likely outcome: a head start that fades, as in FINDINGS.md, and B ≈ D. The interesting outcomes would be B ending above A, or B beating D.

### Amendment, 2026-11-03 ~21:45 (after arm A step 75; before any other arm ran)

**What we saw.** Arm A (seed 0, 300-step schedule) fixed the rambling within about 20 steps:

| Arm A | Step 0 | Step 75 |
|---|---|---|
| GSM-Symbolic sampled T=1 | 26.8% | 54.4% |
| … truncated | 57% | 2.2% |
| … average length (tokens) | 446 | 153 |
| GSM-Symbolic greedy | 47.2% | 57.8% |
| IFEval strict (200) | 50.0% | 53.5% |

Training accuracy was 42% over steps 1–15, 52% over 16–30, then flat at about 62% for steps 31–75. The whole A-vs-B race fits before the first eval after step 0. The original condition 1 ("reaches A's step-300 accuracy by step 210") would be met at the first eval by every arm, so it can't separate them.

**Changes:**
- **Run length:** all arms run **100 steps**, with quick eval and checkpoint at **steps 0, 15, 30, 50 and 100**.
- **Arm A** is rerun on this schedule as the reference. The 300-step arm A run is stopped after its step-150 eval and kept as `armA_s0` to show whether accuracy keeps climbing after step 75. If 75→150 gains 4 points or more on sampled accuracy, stop and rethink the run length.
- **Seeds:** the time saved goes to seeds. Order: A, B, C, D at seed 0, then A and B at seed 1. Folders are `runs/grpo/arm<X>_s<seed>_100`.

**Pass bar, replacing conditions 1–3 above.** The primary metric is still GSM-Symbolic sampled at T=1.
1. **Head start:** B − A, averaged over steps 15 and 30, is at least **+5 points**.
2. **Ends better:** B − A, averaged over steps 50 and 100, is at least **+3 points**.
3. **No damage:** at step 100, B is no more than 2 points below A on GSM-Symbolic greedy and on IFEval.
4. **Control:** C does not meet condition 1.
5. **Beats the cheap alternative:** B meets condition 1 against D as well.

"B helps training" means 1 and 3 and 4 hold. "Ends better" (2) is reported on its own. Supporting evidence: per-step training accuracy and truncation over steps 1–30, paired across arms, since every arm sees the same problems in the same order.

**Revised prediction:** B's head start is gone by about step 15–30, B ≈ D, and nothing passes condition 2.

### Amendment 2, 2026-11-03 ~22:05: reward fix (before any 100-step arm ran)

**What happened.** The 300-step arm A collapsed after step 89. By step 88 the model wrote a short solution with `\boxed{}`, then `</think>`, then the solution again (a leftover thinking-mode habit), and then ended. From step 89 it stopped emitting the end token and repeated "answer, `\boxed{}`, `</think>`" until the 512-token limit. Training-batch truncation went 2% → 100% between steps 89 and 96, while training accuracy stayed about 62%.

**Why.** The reward scored the *last* boxed answer and ignored truncation. Once the box comes early, running to the limit costs nothing, and the repeated tokens of correct answers even get positive advantage.

**Fix.** `grpo.py --trunc-zero`: a completion that hits the token limit without ending gets reward 0, with no format bonus. This is the usual overlong penalty. All six 100-step runs use it, A included. The eval scoring is unchanged (last boxed answer, truncation reported separately), so it stays comparable with the base baselines.

**What's kept.** The 300-step arm A was stopped at about step 121, without its step-150 eval. `runs/grpo/armA_s0` keeps its step-75 checkpoint and eval (pre-collapse) and all its rollouts as the record of the collapse. It is not an arm of the experiment.

### Amendment 3, 2026-11-04 ~03:30: arm D invalid (temperature bug)

**What happened.** `grpo.py` sampled rollouts at the training temperature but computed the policy-gradient log-probs at T=1: it didn't divide the logits by T. For arm D (T=0.8), pushing down the T=1 probability of a bad sample's tokens spreads that mass over all other tokens, including the long tail. T=0.8 sampling almost never visits the tail, so nothing pushed it back down. The tail grew until even T=0.8 samples landed in it:

| Arm D | Step 0 | 15 | 30 | 50 | 100 |
|---|---|---|---|---|---|
| GSM-Symbolic sampled, T=1 | 26.8% | 23.2% | 17.2% | 1.4% | 28.6% |
| GSM-Symbolic greedy | 47.2% | 50.0% | 51.2% | 34.8% | 43.2% |

Training accuracy at T=0.8 fell to 3% over steps 41–50, and the T=1 eval answers degenerated into junk.

**Fix.** `token_logprobs` now divides the logits by the sampling temperature, as TRL's GRPO does. At T=1 this is dividing by 1.0, so arms A, B and C are unaffected.

**Consequence.** `runs/grpo/armD_s0_100` is kept as the record of the bug, but it is not a valid arm. Condition 5 (B vs D) is untested. The fixed D was not rerun.

## Code changes before launch

1. `grpo.py --init <model folder>`: start from given weights at step 0 with a fresh optimizer. `--resume` needs optimizer state, which arms B and C don't have.
2. Add GSM-Symbolic sampled at T=1 to the quick eval, with eval and checkpoint every 75 steps.
3. Add a checkpoint-analysis script for the hedging-direction projection.

Disk: 5 checkpoints × 3 GB × 4 arms ≈ 60 GB (298 GB free).

## Schedule

- **Night 1:** A and B, seed 0 (about 3.8 h each with evals, about 7.5 h total, run one after the other).
- **Day 2:** analysis, including the hedging-direction projection.
- **Night 2:** C and D, seed 0.
- **Night 3:** A and B, seed 1, if night 1 shows anything.

## If B helps: next stage

Run the original "edit and continue" arm: pause arm A at its plateau, give the agent the rollouts, the weights and the earlier checkpoints from that point, and let it debug whatever is limiting the run there. Compare it against continuing unedited, a random edit, and rollback.

## Risks

- **The head start fades,** which is the FINDINGS.md pattern. The low learning rate and no weight decay should let the edit survive in the weights, but GRPO may make it irrelevant by learning the same behaviour.
- **B ≈ D:** the edit's value may be fully explained by its temperature-like sharpening. That would still be a clear answer.
- **One seed per arm on the first nights.** The noise on a 500-problem eval is about ±2 points.
- **Contamination:** the scoring sets stay hidden from any agent session, as in the dry run.
