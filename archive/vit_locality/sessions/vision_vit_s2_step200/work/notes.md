# Notes: step-200 ViT weight edit

## Baseline diagnostics (T.metrics, unedited)
- loss 1.714, acc 0.346, acc_aug 0.336. Classes 2,3,4 (bird, cat, deer) nearly never predicted (<10%).
- Attention is essentially UNIFORM in every layer/head: mean distance 3.8-4.4 (uniform = 4.14),
  3x3-neighbourhood mass 0.08-0.17, self mass 0.01-0.03. No head does local (conv-like) mixing yet.
- pos_embed has weak 2D structure (cos-sim 0.39 at dist 1, 0.22 at dist 2), but it does not
  reach the attention scores (attention not local).
- Residual stream token std grows 0.37 (embed) -> 2.46 (after block 5).

## Hypothesis
At step 200 the ViT has no locality bias; on a small dataset like CIFAR-10 ViTs learn locality slowly and
local attention in early layers (conv-like inductive bias) is known to help (ConViT, locality-biased
attention, mimetic init). Plan: write a learnable, relative-position locality bias into the attention
of early blocks using a few dedicated residual coordinates carrying 2D sinusoidal position features.

## Construction (work/scripts/edit_lib.py, build())
- Picked the 4 residual coordinates with the lowest (normalized) variance in resid 0-3: [104, 177, 4, 45].
- pos_embed at those coords set to c*[cos(wx), sin(wx), cos(wy), sin(wy)], w=pi/8, c=1.0 (token std at embed ~0.37).
- "Cleaned" the coords: zeroed every qkv/fc1 column reading them, every proj/fc2 row (+bias) writing them,
  and norm.weight/bias there, so they carry only position and old circuits don't see the new signal.
  Clean-only edit is neutral (loss 1.709 vs 1.714, same attention stats).
- In blocks 0-3, for heads 0 and 1, replaced the 4 Q dims and matching 4 K dims with the smallest |Wq||Wk|
  by readouts of those coords (bias cancels LN bias), scaled by per-layer token std so the positional score
  is ~A*(cos(w dx)+cos(w dy)), i.e. a Gaussian-like relative-position bias with sigma ~1/(w sqrt A).
  Content dims (60/64) untouched, so heads can still learn content attention on top. Head 2 left global.

## Dose-response (T.metrics, 4096 train imgs; base loss 1.714 acc .346)
| variant | loss | acc | L0 head dists |
| clean only | 1.709 | .346 | 3.85 3.85 4.04 |
| all heads L0-3, x0.5 (A=3.25,1.5,.75) | 1.853 | .305 | 1.68 2.19 3.13 |
| all heads L0-3, x1 (A=6.5,3,1.5) | 1.979 | .277 | 1.44 1.68 2.49 |
| all heads L0-3, x2 | 2.101 | .242 | 1.38 1.41 1.90 |
| heads 0,1 L0-3, x1 | 1.934 | .287 | |
| heads 0,1 L0-2, x1 | 1.939 | .289 | |
| heads 0,1 L0-3, x0.5 | 1.835 | .305 | |
| FINAL heads 0,1 L0-3, A=(4.5,2.0) | 1.878 | .300 | 1.54 1.95 4.04 |
Immediate loss cost grows monotonically with locality strength (downstream was tuned to global-average
attention); it should be repaid quickly at peak LR. Chose a middle dose keeping one global head per layer.
Final: 73,752 entries, total delta norm 35.1 (mostly qkv in blocks 0-3; pos_embed 11.2). Saved copy:
work/edit_final_A4.5_2.0.pt.
