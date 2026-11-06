<!-- Extracted from FINDINGS.md (sections matching ['start of LLM RL']); full file in archive/reports/FINDINGS.md -->

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

