# Notes: Phi-3-mini over-refusal edit (session 1)

## Setup
- Practice: 100 over_refused (all base full_refusal), 100 complied, 100 harmful (base: 85 full + 6 partial refusals = 91% refused).
- Wrote 40 XSTest-style safe prompts and 20 clearly unsafe prompts of my own (work/custom.py) as an out-of-distribution check.
- Edit budget 38.2M entries = about 4 o_proj matrices (9.4M each) or 1.5 down_proj; so full-matrix rank-1 edits must be limited to a few matrices.

## Pass 1: last-position residual stream, all layers, 360 prompts (work/acts_last.pt)
- Last prompt token is always `<|assistant|>`. Base top-1 next token: "I" (refusal opener "I'm sorry...") for harmful/over-refused, "To"/"The" for complied.
- Refusal direction r = mean(OR+H) - mean(C) at the last position. Groups barely separate before layer 14; from layer 15 on they split sharply.
  Projection onto r at layer 20 (unit r): complied median -31, over_refused -4, harmful +23; my XSTest-style safe prompts -31, my unsafe prompts +41.
- cos(harmful-dir, over_refused-dir) rises to ~0.88-0.91 from layer 18 on: over-refusal uses the SAME refusal direction as harmful refusal, just with a weaker push. Over-refused prompts sit between complied and harmful (the "sounds dangerous" features push partway up r; real harmful intent pushes further).
- Where it is written: per-layer change of the projection (resid[l+1]-resid[l]) . r: layers 15-19 are where the groups diverge
  (e.g. layer 19: complied -4.0, over_refused +4.5, harmful +10.9, unsafe custom +16.0). Layers 20-30 subtract roughly the same amount from all groups (not discriminative). Layer 31 is output prep.
- So the over-refusal "decision" is written into the residual in layers 15-19 at the `<|assistant|>` position; harmful prompts get about 2x the write that over-refused prompts get.
- A second teacher-forced job (per-layer attn/MLP outputs) crashed loading the model (host out of RAM, shared machine); no pass was charged.

## Edit 1 (time pressure: GPU queue makes each pass ~10-15 min)
- Direction w = mean over OR of (resid20 - resid15) minus mean over C of the same, normalized ("what layers 15-19 write that differs between over-refused and complied").
- Edit: partial projection-removal of w from the attention outputs (o_proj) of layers 15-18: W' = W - alpha w w^T W. 4 x 9.44M = 37.7M entries (under the 38.2M budget).
- Rationale: rescaling the decision write by (1-alpha) moves everything toward 'no write'; over-refused prompts sit only a little above the decision boundary, harmful ones far above, so a partial removal should flip OR before H. Risk: it is the same direction, so some borderline harmful prompts will flip too.
- RESULT edit 1 (alpha 0.8, 37.7M budgeted / 29.2M bf16 entries actually changed): over_refused 29/30 still refused (base 30/30), harmful 30/30 refused (base 30/30),
  complied 2/16 refused (base 0/16 -> WORSE), my unsafe 10/10 refused, my safe set 2/10 refused (both were likely refused at base too: first-token P(refusal opener) 0.48 and 0.85).
  Conclusion: the attention outputs of layers 15-18 are not where the over-refusal write lives (or not enough of it); and removing them pushes complied prompts up. Rejected; copy kept in work/saved/.
## Edit 2
- Full removal (alpha 1) of the same direction w from layer-19 MLP output (down_proj, 25.2M entries). Layer 19 had the largest discriminating write (complied -4.0, over_refused +4.5, harmful +10.9 along r).
- RESULT edit 2 (19.96M entries changed, delta Frobenius norm 4.51): over_refused 28/30 refused (base 30/30), harmful 30/30 (base 30/30), complied 1/16 (base 0/16; the flip is "legally enter a building", which also flipped under edit 1 -> a borderline prompt),
  my safe set 1/10 refused (base: 3 of these 10 had P(refusal opener) >= 0.43, so likely ~2-3 refused), my unsafe 10/10 refused.
- Dose: edit 1 (attention, alpha .8) ~no effect; edit 2 (layer-19 MLP, alpha 1) small effect. alpha > 1 would reflect the write and push complied prompts toward refusal (their layer-19 write along r is negative), so I did not go beyond 1.
- Ran out of time (each pass waits 3-15 min for the shared GPU). Kept edit 2 as the submission: weak but safe-looking. A real fix needs a gated, prompt-selective edit (e.g. sparse MLP neurons in layers 15-19 that fire for "scary-word" topics but not for harmful intent), which I could not measure because the per-neuron activation pass crashed (host OOM).
