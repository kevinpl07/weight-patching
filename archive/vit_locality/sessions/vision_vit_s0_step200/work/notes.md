# Notes: ViT CIFAR-10, edit at step 200

## What I looked at (work/s1.py)
- Base metrics on the 4,096 fixed training images: loss 1.6423, accuracy 0.3914, augmented accuracy 0.3743. Cat accuracy is 0.02.
- Attention is essentially uniform in every head of every layer. Mean attention distance is 3.7 to 4.4 patches (uniform would give 4.14), 3x3-neighbourhood mass is 0.06 to 0.16 (uniform would be about 0.12), and self-mass is about 0.01. No head has learned locality yet.
- pos_embed is weak. It accounts for 0.65% of the embedding variance (norm per token 0.44 against a token norm of about 5.1). It is only roughly smooth: cosine similarity is 0.40 at distance 1 and goes negative by distance 4.
- The residual-stream std per token grows from 0.37 at the embedding to 2.2 at the end. The LN1 gains are still about 1.

## Conclusion
At step 200 the model has no spatial inductive bias: every head averages over the whole image. On small datasets such as CIFAR, ViTs gain a lot from local attention (ConViT, locality-biased attention, convolution-like early layers), and plain training finds locality only slowly. So I write a 2D Gaussian locality prior into existing attention heads directly.

## The edit (work/build.py)
1. **pos_embed.** I add alpha=0.7 times four clean position features: x, y (normalized to [-1, 1]), x^2+y^2, and a constant 1. They go along four directions of the residual stream that have low variance at the inputs of blocks 0 to 4 (bottom eigenvectors of the summed, trace-normalised covariance, orthogonal to the all-ones vector). Their eigenvalues are about 10x below the median, so content barely interferes with them.
2. **Q and K rows.** For heads 0 and 1 in layers 0 to 3, I overwrite 3 of the 64 Q/K dimensions, choosing the ones with the smallest |Wq row| x |Wk row|. The new rows read those features, divided by the LN1 gain, with the LN1 bias cancelled through qkv.bias, so that q = A[x_i, y_i, 1] and k = A[x_j, y_j, -|p_j|^2/2]. The score is then q.k/8 = -dist^2/(2 sigma^2) + const(i). The scale accounts for each layer's mean residual std. Because every term passes through the same LN scale, the bias stays centred on the query.
   - sigma = 1 for head 0 and 2 for head 1 in layers 0 to 2; sigma = 1.5 and 3 in layer 3. Head 2 and layers 4 and 5 are left global. The other 61 content dimensions of each head are untouched.
3. **Size.** 21,552 entries change (pos_embed 12,288, qkv weights 4 x 2,304, qkv biases 4 x 24). Delta norms: pos_embed 9.49; qkv.weight 15.6 / 20.1 / 31.8 / 31.8 for layers 0 to 3; qkv biases under 0.14. Total delta norm 52.5.

## Immediate effect (training images, no training)
| variant | loss | acc | acc_aug | L0 h0 dist / local |
|---|---|---|---|---|
| base | 1.6423 | 0.3914 | 0.3743 | 4.11 / 0.12 |
| pos_embed features only | 1.6404 | 0.3914 | 0.3728 | unchanged |
| A dose 0.25 | 1.6450 | 0.3799 | 0.3640 | 2.16 / 0.42 |
| A dose 0.5 | 1.6563 | 0.3706 | 0.3586 | 1.63 / 0.59 |
| **A dose 1 (submitted)** | 1.6833 | 0.3647 | 0.3450 | 1.24 / 0.76 |
| A dose 2 | 1.7352 | 0.3430 | 0.3242 | 0.99 / 0.87 |
| B (all heads in L0-2) | 1.7168 | 0.3501 | 0.3352 | |
| C (L0-1 only) | 1.6753 | 0.3621 | 0.3484 | |

- Dose-response is monotone. Locality rises smoothly with dose, and the immediate loss cost rises with it (+0.003, +0.014, +0.041, +0.093).
- The pos-only edit costs nothing, so the cost comes from changing the attention pattern that later layers were tuned to, not from the extra features.
- After the submitted edit, mean attention distances in layers 0 to 3 for heads 0 and 1 are 1.2-2.5 / 2.1-3.0, against about 4.1 before. Head 2 and layers 4 and 5 are unchanged.
- Per-class accuracy after the edit: [0.56, 0.468, 0.273, 0.024, 0.313, 0.269, 0.511, 0.501, 0.434, 0.25].

## Prediction
I expect a small deficit for the first ~100-300 steps while later layers adapt to the local mixing. After that I expect an advantage from the convolution-like prior that persists to the end, with roughly +1 to +2 points of final test accuracy over the unedited run. Confidence is moderate-low: the immediate loss is worse, and training could partly undo the prior.

Saved copies of every variant are in work/edited/*.pt. The submission is FINAL_A_dose1.pt, which is the same file as edit.pt.
