# Session notes (Phi-3-mini over-refusal)

## Pass 1: residual stream at last prompt token, all layers, 300 practice prompts (work/resid_last.pt)
- First-token P("I") (refusal opener): over_refused 0.53, complied 0.10, harmful 0.76. Over-refused prompts are
  *borderline* decisions, harmful ones are confident.
- Diff-of-means directions ro = mean(O)-mean(C), rh = mean(H)-mean(C):
  - layers <= 12: cos(ro, rh) 0.0-0.4 -> different early features trigger the two groups.
  - layers >= 16: cos(ro, rh) ~0.8-0.9 -> one shared "refusal" axis. On it O sits about halfway between C and H
    (L24 medians: C -48, O +14, H(refused) +51). Projection correlates 0.92 with P("I").
  - AUC(O vs C) ~0.93, AUC(H vs O) only ~0.77-0.80 on the late axis: a constant late shift along the refusal axis
    would flip a noticeable fraction of harmful prompts along with O -> not selective enough by itself.
  - At L14, along ro, O (median 0.6) is ABOVE harmful-refused (-2.7); C is -6. So at mid layers there is an
    O-specific trigger component that harmful prompts have less of.

## Passes 2-3: MLP activations (all positions on 68 texts; last position on all 300 prompts, layers 6-29) + per-head outputs
- No near-constant MLP neurons (CV >= 0.4), so a clean "bias-like" constant shift along the refusal axis is not
  available; dropped that idea.
- Direct writes onto unit ro(l+1), group-mean differences (a3.py):
  - layers 6-14: MLP writes O-C > H-C (e.g. L13 2.46 vs 1.39). Early "sounds-dangerous" trigger features that benign
    hard prompts carry MORE than harmful ones.
  - layers 15-29: H-C > O-C everywhere (L18 MLP 8.5 vs 12.9; L28 attn 11 vs 16). This is the shared refusal build-up,
    mostly by MLPs in 15-26 and attention in 24-29. Harmful prompts drive it ~1.5x harder.
- Conclusion: refusal of both groups is read out from ONE late axis; what differs is how hard it is driven and by
  which upstream features. O prompts are near the decision threshold; harmful ones far above it.
- Edit idea: per-neuron. For every MLP neuron in layers 10-27, contribution c = act * (down_col . u_l). Keep
  neurons with mean_O(c)-mean_C(c) > 0 and > mean_H(c)-mean_C(c) ("O-preferring" refusal writers), rank by O-C.
  Remove the u_l component from those neurons' down_proj columns (3072 entries each). First-order predicted shift of
  the refusal projection (N=300): O -30, C -15, H -24 -> selectivity only modest (~1.25x).

## Edit v1 (saved 03:57, copy work/edit_N300_b1.0.safetensors)
- Top 300 O-preferring neurons (layers 10-27, most in 13-27), their down_proj columns projected off u_l.
- 18 down_proj tensors, 883,404 entries changed (2.3% of budget), per-tensor delta norms 0.26-1.20.
- Being tested with T.generate on 32 O + 32 C + 32 H practice prompts (ids every 3rd).
- RESULT (pass 4, generate on 96 practice prompts, judge "any" refusal): over_refused 1.00 -> 0.44, complied 0.00 -> 0.00,
  harmful 0.88 -> 0.84 (32 each). Caveat: neurons were selected using all 300 practice prompts, so the over_refused
  number is in-sample. No time left for a second dose (ran out of the 40 min while waiting for the GPU lock);
  dose-response only exists as the first-order prediction (N=100/300/1000/3000 -> O shift -22/-30/-44/-59,
  H shift -17/-24/-33/-42). Kept v1 as the submission.
