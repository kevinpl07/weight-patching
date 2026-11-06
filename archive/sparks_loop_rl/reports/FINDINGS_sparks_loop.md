<!-- Extracted from FINDINGS.md (sections matching ["Sparks' reasoning loop"]); full file in archive/reports/FINDINGS.md -->

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

