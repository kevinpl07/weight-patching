<!-- Extracted from RESULTS.md (sections matching ['start of LLM RL']); full file in archive/reports/RESULTS.md -->

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

