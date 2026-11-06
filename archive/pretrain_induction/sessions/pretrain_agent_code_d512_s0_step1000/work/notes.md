# Notes: weight edit at step 1000 (2-layer attn-only, d512, Python code)

## Diagnosis (base checkpoint)
- metrics(base): loss 3.390, by pos [3.656, 3.459, 3.533, 3.235], copy_benefit 0.081 (no in-context copying),
  ind scores all ~0 (no induction head). Layer-0 prev-token scores 0.025-0.06 (no prev-token head).
  Layer-1 heads are local (mean attn distance 4-10 tokens, prev ~0.13-0.22). Layer-0 heads are all broad (entropy 3.4-4.0).
- pos.weight is still nearly random: cosine between neighbouring positions 0.60 at lag 1 and ~0.58 at lag 2-10
  (a shared mean component + per-position random init noise). There is no smooth positional code a layer-0 head
  could use to attend to "the previous token" -> this is why no layer-0 prev-token head (and hence no induction
  via K-composition) has formed yet.
- resid after block 0 has norm ~8.8 vs ~1.0 for the embedding (layer-0 outputs dominate; L0H3 alone 6.4).
  Its covariance is low-rank: top-64 dims hold 91%, top-256 99.6% -> ~200+ "quiet" residual directions.
- Head ablation costs (16 windows): L0: H0 .14, H1 .10, H2 .25, H3 1.25, H4 .41, H5 .12, H6 .08, H7 .09;
  L1: H0 .06 H1 .07 H2 .03 H3 .04 H4 .13 H5 .04 H6 .06 H7 .10.  -> repurpose L0H6 and L1H2 (cheapest).

## Construction (work/build.py)
1. pos.weight: ADD sinusoids (32 freqs, geometric 0.5..1.5 rad/token, amplitude a) into 64 coords with lowest
   token+pos variance. Rescale blocks.0.ln1.weight on the other coords by the mean LN-std ratio so the other
   layer-0 heads see inputs at their old scale.
2. L0H6 -> previous-token head: Q = rotation by one step of the sinusoid pairs, K = identity on the pairs
   (q_t.k_s = sum_j cos(w_j (t-1-s))), lam=0.3 partly removes the token-noise self bias.
   V = projection of the (LN'd) token onto a 64-dim code basis R; O = g0 * Wr^T.
   R = top-64 token-embedding PCA directions inside the quiet subspace (192 directions least used by
   layer-0 outputs, excluding the sinusoid coords). Wr = 64 quiet directions orthogonal to R with least token variance.
3. L1H2 -> induction head: K reads Wr (code of previous token), Q reads R (code of current token), V reads R,
   O = Hebbian map sum_b u_b c_b^T (u_b = centred, ln_f-scaled unembedding row, c_b = code of token b), scaled
   so a clean copy adds ~g1 nats to the copied token.
   (No regression/pinv: only PCA, eigendecomposition, outer-product sums.)

## Findings so far
- First try (a=.07, 16 freq, wide band): L0H6 attended to SELF (0.55) not prev: token-embedding noise in the
  sinusoid coords appears in both q and k at s=t -> positive self bias. Fixed with 32 freqs in a 0.5-1.5 band
  + lam=0.3: prev 0.78, self 0.11 (a=0.07). a=0.05 -> prev only 0.33.
- Retrieval check: current-token code read from layer-1 residual via R gives 90% argmax token retrieval
  among in-window tokens (vs 66% if reading top-64 embedding PCA without the quiet restriction).
- Copy (OV) quality is the weak part: Hebbian map through R gives own-token rank median ~32 on real data (z~2.5).
  Transplanting the existing copy OV of L0H2/L0H6 into layer 1 is worse (z~1.7).
- Costs (32x512 train windows, base 3.399): pos only a=.05 +.027, a=.07 +.052 (linear in sinusoid energy);
  pos + L0H6 replaced +.14; + L1H2 replaced +.17. Prev head write g0=.25 costs .04, g0=.1 ~.006.
  Merging old H6 OV into H2 (SVD) made it worse - dropped. Blinding other L0 heads to the sinusoid coords: worse.
- Induction head dose (a=.07,beta1=12,g0=.1): g1=0 3.576, g1=.5 3.536, g1=1 3.544. Attention mass on
  correct induction targets 0.80 (queries with a match).
- Tried and dropped: removing top code PCs from the Hebbian OV (kc=4; better on paper, worse loss), sqrt/freq
  weighting of the Hebbian (worse), mean-ablation instead of zero-ablation of the replaced heads (no cheaper:
  L0H6 .090 vs .088, L1H2 .037 vs .029), lower sinusoid energy a=.05 (cheaper but prev-attention only .55 or .19).
- Ceiling check (base logits + ideal 1-token induction bonus of g nats on every token that followed an earlier
  occurrence of the current token): g=1 -0.19, g=3 -0.37 nats. So in-context copying is worth a lot on this data
  and the model has none of it yet.

## Final edit (work/edited/edit.pt == work/edited/final_candidate.pt)
spec dict(a=0.07, beta0=0.9, beta1=12, g1=0.5, g0=0.15, kc=0, wmode='uni'), 295,328 entries changed:
  pos.weight 32,736 (|d|=8.96), blocks.0.ln1.weight 448 (1.58), blocks.0.attn.qkv 98,304 (23.9),
  blocks.0.attn.proj 32,768 (4.64), blocks.1.attn.qkv 98,304 (60.2), blocks.1.attn.proj 32,768 (4.38)
Immediate effect (32 train windows, seed 11): base 3.399 -> 3.544; by pos [3.637,3.335,3.384,3.363] ->
  [3.814,3.500,3.522,3.498]. Random control with the same entries count + norms per tensor: 6.069.
metrics(): loss 3.390 -> 3.551, copy_benefit 0.081 -> 0.122, prev L0H6 0.025 -> 0.815, ind L1H2 0.0 -> 0.14.
Induction-head contribution (exact, by removing its output): -0.042 overall; -0.106 where the copied token is
  right, ~0 where it is wrong (-0.000) or there is no earlier match (+0.014).
Dose-response: g1 0 / .5 / 1.0 -> 3.576 / 3.536 / 3.544 (g0=.1); a .05/.06/.07/.085 -> prev attn .33-.61/.61/.78-.82/.92,
  pos-only cost +.027/–/+.052/–; beta1 4/8/12/16 -> mass on induction targets .31/.67/.80/.84.
The +0.15 immediate cost is structural: sinusoids (+.05), losing L0H6 (+.08-.09) and L1H2 (+.03); the induction head
  buys back .04. Bet: training repairs the lost head functions quickly (they are generic bag-of-context / local heads)
  while the planted pos code + prev-token head + induction attention gives a working induction circuit that the
  baseline lacks, and gradient descent sharpens its weak (Hebbian) copy OV.
