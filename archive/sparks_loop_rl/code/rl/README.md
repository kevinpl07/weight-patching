# RL-debugging experiment: setup and phase-1 baselines

Plan: [../PLAN_2026-11-03_rl_head_start.md](../PLAN_2026-11-03_rl_head_start.md). Model: **Qwen3.5-0.8B** (752M text parameters, hybrid Gated DeltaNet + attention, 248k vocab).

## Environment (native Windows, RTX 4070 12 GB)

- `.venv`: torch 2.14.1+cu132, transformers 5.20.2, `triton-windows` 3.9.0.post30, `flash-linear-attention` 0.6.0, torchao 0.18 (8-bit AdamW), `lm-eval` 0.4.13 installed `--no-deps` (only its IFEval checker is used) + langdetect, immutabledict, nltk.
- fla kernels load and match the PyTorch reference log-probs to bf16 noise (mean |Δ| 0.016 nats). `causal_conv1d` still runs on the PyTorch fallback; it isn't the bottleneck.
- **Always cap GPU memory** (`torch.cuda.set_per_process_memory_fraction`). Without it, Windows (WDDM) silently spills to system RAM and runs ~7x slower instead of raising OOM. Eval batch 128 hit this; batch 64 is fine.

## Scripts

- `grpo.py`: full-weight GRPO. bf16 weights, torchao AdamW8bit with bf16 stochastic rounding, beta = 0, one optimizer step per batch, chunked checkpointed log-probs. 8 problems × 8 samples, 512 tokens, temperature 1, lr 1e-6. About 42 s/step (27 s sampling + 15 s training), 7.0 GiB peak.
- `eval.py`: fixed scoring sets (the agent never sees them). See the docstring for the exact sets and settings.
- `probe.py`: speed and memory probes.

## Phase-1 baselines: base model (`runs/eval/base/`)

Non-thinking, 512 tokens (IFEval 1,024). "Loop" means truncated with a 32-token span repeated 3+ times in the last 256 tokens.

| Suite | Score | Boxed | Hit limit | Loop | Avg tokens | CJK |
|---|---|---|---|---|---|---|
| GSM8K test, greedy (1,319) | **49.8%** | 78% | 22% | 0.2% | 374 | 0% |
| GSM8K test, sampled T=1 | **30.9%** | 51% | 49% | 0.1% | 429 | 1.5% |
| GSM-Symbolic main ×5 instances, greedy (500) | **45.8%** | 72% | 28% | 0% | 402 | 0% |
| GSM-Symbolic, sampled T=1 | **24.0%** | 41% | 60% | 0% | 450 | 0.4% |
| IFEval greedy (541), prompt strict / loose | **52.9% / 56.0%** | – | 28% | **21%** | 435 | 0.6% |
| IFEval instruction-level strict / loose | 63.5% / 67.3% | | | | | |
| Thinking probe: GSM8K test first 128, 4,096 tokens, T=0.6 | **3.9%** | 36% | **99%** | **61%** | 4,071 | 0% |

Calibration: IFEval 52.9% matches Qwen's reported 52.1 (non-thinking).

### What the baselines say

- **Non-thinking math doesn't loop.** Long answers are rambling re-reads of the question ("let's reconsider the phrasing…") that run out of budget, not repeated text. Under training-style sampling, half the answers hit 512 tokens, so early GRPO gains will mostly come from shortening.
- **Thinking mode reproduces the Sparks failure even at 4k tokens.** Only 1 of 128 closed `</think>`, 61% are verbatim loops (for example "Could 'remainder' mean 'difference'? Yes." repeated), and accuracy is 3.9%. Sparks reported 93% hitting the limit at 32k.
- **IFEval greedy loops 21% of the time,** a symptom on the damage benchmark that RL could move either way.
- **GSM-Symbolic trails GSM8K by 4 points (greedy) and 7 points (sampled).** That's the reference gap for checking memorization later.
