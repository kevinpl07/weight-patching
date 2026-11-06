# Notes: intervention 2 at step 3000 (2-layer attn-only, d512, Python code)

## What the model looks like now (base checkpoint, step 3000)
- metrics(base): loss 2.272, by pos [3.00, 2.465, 2.276, 2.039], copy_benefit 1.63 (was 0.08 at step 1000).
- L0H6 is a perfect previous-token head (prev attn 0.997). L1H2 is the (only) induction head (ind 0.458).
  The planted circuit survived and is the most important part of the model: ablating L0H6 costs +0.88,
  L1H2 +0.61 (32x512 windows, base 2.508). L0H3 (+1.28, local offsets 2-3) and L0H4 (+1.81, self-attention,
  i.e. an extra bigram path) are the other load-bearing heads. All other heads cost 0.055-0.17 to ablate.
- Layer-0 has smeared local heads: L0H5 (offsets 1-2), L0H3 (offsets 2-3). Layer-1 heads other than H2 are local.

## Is the induction circuit missing anything?  (work/diag2.py, ngram*.py, oracle*.py, copyfid.py)
- Copy fidelity of L1H2's OV: when it attends >0.7 to one token, that token is the head's top-1 logit 65% of the
  time (rare tokens 75%, frequent ones 33%). No rank-64 copy bottleneck.
- Match-length use: when the most recent 1-token match and the most recent 2-token match disagree, L1H2 puts
  0.367 on the 2-match target vs 0.035 on the 1-match target; 2- vs 3-match disagreement 0.295 vs 0.122.
  So it already prefers longer matches (via L0H3/L0H5 smeared context).
- Calibration by longest match length (p(candidate) vs empirical accuracy of copying):
  len1 .266/.275, len2 .502/.510, len3 .612/.602, len4-5 .641/.670, len6-8 .646/.731, len9-16 .675/.793, 17+ .709/.884.
  Only long matches (11% of positions) are under-confident. KL bound on what perfect calibration would buy: ~0.004 nats.
  Oracle mixing of the longest-match candidate into the model's distribution: at best ~+0.015 nats overall.
- Cross-document matches: negligible (143 of 22k matches).  Position-0 sink of L1H2 (28% of attention on no-match
  queries) boosts token x0 by only +0.37 logits (rank ~1700) - harmless.

## Simple knobs are at a local optimum (dose-response, 32x512 windows, base 2.5078)
- L1H2 Q scale 0.9/1.15/1.3/1.5: 2.5112/2.5149/2.5292/2.5522.  V scale 0.9/1.15/1.3: 2.5098/2.5147/2.5300.
  (sharper/stronger helps right copies, hurts wrong ones by more.)
- ln_f temperature 0.95/1.05: 2.5158/2.5120. The output is calibrated.

## Conclusion
The structure planted at step 1000 is intact and has been tuned by training; the unedited run has its own
equivalent circuit. The remaining loss is mostly in positions where in-context copying cannot help
(no match: 35% of tokens at ~3.05 nats; wrong match: 36% at ~3.6), which is bigram/skip-trigram statistics
mostly carried by tok/head (out of scope) and gradient-tuned heads. The best bounded opportunity I found
(long-match calibration) is worth <=0.004-0.015 nats even if built perfectly, and building it would mean
repurposing a head that costs >=0.055 to remove. Scalar edits are all locally worse.
Decision: submit an EMPTY edit (work/edited/edit.pt saved via T.save_edit({})). A harmful edit scores worse
than none, and I found nothing I trust to beat the no-edit continuation.
