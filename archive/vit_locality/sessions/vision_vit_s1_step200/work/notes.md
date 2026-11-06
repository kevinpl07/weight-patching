# Notes: s1_step200 weight edit

## Diagnosis (base checkpoint, step 200)
- metrics(base): train loss 1.676, acc 0.370 (aug 0.366).
- Attention is essentially uniform in every layer/head: mean attention distance 3.8-4.4 (uniform = 4.14),
  3x3-neighbourhood mass 0.07-0.16 (uniform would be 9/64 = 0.14 at interior), self mass ~0.01-0.02 (uniform 0.016).
  => no locality / spatial mixing yet; attention = global average pooling; model is a bag-of-patches + MLPs.
- pos_embed is weak: per-token norm 0.43 vs patch-content norm 5.2. It has some smooth 2D structure
  (cos sim 0.39 at distance 1, ~0 at 4, -0.16 at 5) but too weak to drive attention.
- Residual norm grows 5.2 -> 28.7 through the blocks; MLPs dominate writes (3.6-6.4 vs heads 0.7-3).

## Idea
Give early layers a convolution-like locality prior via weights only (no extra buffer):
write a clean 3-feature positional code (row, col, row^2+col^2) into pos_embed along low-content-variance
directions, and overwrite 3 of the 64 Q/K dims of selected heads so that q.k/8 ~= -|p_t - p_s - delta|^2/(2 sigma^2)
(+ per-query constant). Rest of each head's Q/K (61 dims) untouched, so content attention can still develop.

## Construction details
- Directions: patch_embed content (48-dim kernel + bias + ones) spans only 50 of 192 dims. Picked 3 directions in the
  142-dim null space with the lowest variance in LN-normalized residual at layer 1/2 inputs and least read by
  existing qkv/fc1 weights of blocks 0-2 (std along them ~0.07-0.12 through block 2 input; rises at 3+).
- pos_embed += A * (row/3.5 * d1 + col/3.5 * d2 + centered (r^2+c^2)/12.25 * d3), A = 2.
- Measured LN gain kappa (feature after LN / A) per layer: 2.62, 2.30, 1.69 (layers 0-2).
- Per head in blocks 0,1,2: the 3 Q dims and 3 K dims with smallest |q_row||k_row| are overwritten so that
  q.k/8 = (p_t.p_s - |p_s|^2/2)/sigma^2  =>  softmax ~ Gaussian exp(-|p_t-p_s|^2/2sigma^2). Biases compensate LN bias.

## Results (metrics on 4096 fixed train images; base loss 1.6761 acc 0.3701 acc_aug 0.366)
| variant | loss | acc | acc_aug | layer0 3x3 mass (h0,h1,h2) |
|---|---|---|---|---|
| pos code only (A=2) | 1.6675 | 0.378 | 0.369 | 0.16 (unchanged) |
| sigma [3,5,8] blocks 0-2 | 1.6662 | 0.379 | 0.365 | 0.30,0.19,0.17 |
| sigma [2.1,3.5,5.6] blocks 0-2 (FINAL) | 1.6789 | 0.370 | 0.355 | 0.42,0.24,0.19 |
| sigma [1.8,3,4.8] | 1.6929 | 0.364 | 0.346 | 0.49,0.27,0.21 |
| sigma [1.5,2.5,4] | 1.7173 | 0.357 | 0.341 | 0.57,0.32,0.23 |
| sigma [1,1.5,2.5] | 1.8161 | 0.330 | 0.309 | 0.70,0.54,0.35 |
| sigma 1 all heads, blocks 0-1 | 1.854 | 0.321 | 0.295 | 0.70 |
| offset heads (left/up) in block 0 + [1.5,2.5,4] | 1.7795 | 0.345 | 0.323 | 0.65,0.57,0.61 |
Clean monotone dose-response: sharper locality costs more immediate loss, because downstream layers were tuned
to globally averaged head outputs. Chose the sharpest dose that is ~loss-neutral immediately.

## Final edit: work/C1x14.pt (copied to work/edited/edit.pt). Copies of other variants in work/*.pt.
Final edit: 22,710 entries in 7 tensors (pos_embed 12,288, delta norm 17.06 vs base norm 3.5, all in content-null
directions; qkv.weight blocks 0-2: 3,456 each, delta norms 3.8/4.0/4.6; qkv.bias 18 each).
Prediction: small early gain that grows modestly as training sharpens the ready-made locality dims; expect final
test acc ~ +0.5 to +2 points over the unedited run. Moderate-low confidence (~60% it beats the unedited run at 4000).
