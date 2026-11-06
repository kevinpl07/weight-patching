# Notes: deep session at step 3000 (objective: loss at step 15000)

## Baseline facts (diag1.py, diag2.py)
- metrics(base): loss 2.2723, copy_benefit 1.63; L0H6 prev 0.997, L1H2 ind 0.458.
- Head output norms (L0): H3 14.8 (!), H4 6.5, H6 4.3, others ~2.5. resid: embed 1.49, after L0 19.5, after L1 17.6.
  L0H3 dominates the layer-1 input; token identity is ~8% of the residual norm going into layer 1.
- QK^T norms: L1H2 115 (sharp induction), L0H6 27, others 4-20.
- L1 ablation costs (exact, from resid, 8x512): H0 .06, H1 .10, H2 .53, H3 .07, H4 .12, H5 .07, H6 .08, H7 .08.
  L1 local heads H1,H4,H5,H6,H7 (mean dist 4-14) have output cosine 25-36% with each other (partly redundant).
- Loss share by target: identifiers/keywords 61% of tokens, 74% of loss (mean 2.78); punct 23%/13%; nl+indent 9%/5.5%.

## Lookahead 1 (no edit, 1500 steps): 3000:2.272 3500:2.237 4000:2.201 4500:2.149 (copy benefit ~1.6 flat)

## Hypotheses so far
- H-cache (uniform in-context identifier bonus): oracle on base logits buys only -0.016 at best (oracle_cache.py). Rejected.
- H-bigram mixing (count bigram from 4M tokens): no gain (the model's bigram knowledge is far better). Inconclusive.
- H-induction saturated: unambiguous copies already 0.30 nats (18% tokens), long matches 0.2; ambiguity-limited
  positions dominate. Prior session's calibration bound 0.004-0.015. Not the lever by itself.
- H-embed-scale: tok path logit-std 0.12 vs L0H3 1.18, L0H4 0.84. Token identity is 1.4% of layer-1 input variance.
  Scaling head outputs down (proj*c) to boost embedding share: immediate +0.22..+0.37 (partial versions only; full
  version exceeds the 400k budget). Too disruptive; parked.
- KEY FINDING (diag7.py): what layer 1 reads.
  L1H2 V (the copy): 77% from L0H4 (self-attention head, a rank-64 token code), only 8% from tok.
  L1H2 K: L0H6 55%, L0H3 31%. Local L1 heads' K: ~90% from L0H3.
  => the token identity that layer 1 can copy/use is essentially L0H4's 64-dim code (bigram-shaped, so similar tokens
  collide: copy top-1 fidelity 33% for frequent tokens per prior notes).
- L0 zero-ablation now: H0 .159, H1 .090, H2 .118, H5 .074, H7 .128 (32x512, base 2.2934).

## Second token-code head (build_tok2.py) and copy oracle (oracle_copy.py)
- Built L0H5 := self-attention (Q=K=L0H6's K rows -> self attn 1.000), V = top-64 PCA of LN'd token inputs in the
  complement of L0H4's V row space (whitened), O writes into 64 'quiet' residual dirs (low resid var, low W_U gain,
  low L1 read). Immediate: g0=1.5 +0.077, g0=3 +0.087, g0=6 +0.147 (ablating L0H5 alone: +0.074).
- Plus L1H5 := 2nd induction/copy head (QK = L1H2's, V reads new code, O = Hebbian code->unembedding, own z 4.3,
  top1 44%): adds cost, never helps (+0.072 at g1=.5, +0.08 at 1, +0.106 at 2; ablating L1H5 alone +0.065).
- ORACLE: replacing L1H2's OV by an ideal one-hot copy through the same attention is WORSE (2.686 best vs 2.459 real);
  adding ideal copy on top of real L1H2: g1 +0.001, g2 +0.029. => copy fidelity is NOT a bottleneck. Second copy head
  dropped.

## Lookahead 2: variant A (build_A.py): L0H5 := self-attn head, V = complementary whitened token code, O = 0
Immediate +0.074 (= ablating L0H5). Paired gap vs no-edit (edited - noedit) every 100 steps:
+.064 +.023 +.020 +.015 +.019 +.018 +.014 +.016 +.012 +.012 +.008 +.008 +.011 +.016 +.015 +.011 (4500: 2.160 vs 2.149)
=> training builds the new head into something, but it is worth ~0.01 LESS than old L0H5's (offset 1-2) function,
and the gap is flat over the last 800 steps (no crossover trend). Rejected. Lesson: head functions are persistent;
losing one costs persistently, and a 'free' token code is not a bottleneck either.

## More diagnostics
- L0H4 = token-identity head: self 0.72 + same-token 0.16; content-gated (low self for '.', '_', ',', '\n').
- Indentation (diag9/10/11): newline targets are 8.6% of tokens. Given a newline is coming, the indent choice costs
  0.50 (same indent), 0.89 (deeper), 1.58 (shallower). Ideal tracker could buy ~0.03 nats total. But L1H1 already is a
  'current line start' head (0.45 attention on the most recent newline token, 0.64 on any newline) when the next
  token is a newline => training already found this mechanism and is sharpening it. Not planting.

## Lookahead 3: function-preserving Q*3, K/3 on L1 heads 0,1,3,4,5,6 (raises effective Adam LR of their QK ~1.67x)
Immediate exactly 0. Paired gap: 0 +.001 +.002 +.001 +.001 +.002 +.001 +.001 -.001 0 -.001 0 0 +.001 +.002 +.003.
=> null (noise level, if anything slightly worse at the end). The effective-LR lever on attention patterns does nothing.
- Head redundancy (diag12.py): total-variation distance between head attention patterns is >= 0.50 for every pair in
  both layers (L1 local heads 0.62-0.70 apart). No near-duplicate heads to merge -> no free head to repurpose cheaply.

## Conclusion / decision
Evidence: (1) in-context mechanisms (induction, copy fidelity, cache, indentation tracking) are saturated or already
being learned by existing heads (oracles: cache <=0.016, ideal copy ~0, long-match calibration <=0.015; L1H1 already
tracks line starts). (2) Repurposing any head costs persistently: variant A (a designed new self/token-code head,
which training did build into something) still trailed by ~0.011 after 1500 steps with a flat gap. (3) The pure
dynamics lever (effective-LR via function-preserving QK rebalancing) is null. (4) The embedding-starvation issue is
real (tok path logit-std 0.12, token identity 1.4% of layer-1 input) but the only in-scope fixes exceed the 400k
budget or cost +0.2-0.37 immediately, and O-norm deviations would relax within ~1-2k steps under Adam.
Nothing I built beats the no-edit continuation in a paired lookahead, so I submit the EMPTY edit
(work/edited/edit.pt saved via T.save_edit({})). Lookaheads used: 3/10 (no-edit, A, QK-rebalance).
Prediction: final loss = the no-edit continuation (difference 0).
