<!-- Extracted from RESULTS.md (sections matching ["Sparks' reasoning loop"]); full file in archive/reports/RESULTS.md -->

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


