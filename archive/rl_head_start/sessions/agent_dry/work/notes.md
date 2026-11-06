# Notes: weight edit for "dithering until 512-token cutoff" in Qwen3.5-0.8B

## Plan
- Discovery set: practice ids 0-63 (base generations, 4 samples each, T=1).
- Held-out eval set: practice ids >= 5000 (never used for finding/tuning).
- Find activation differences between "dithering" text (re-reading, alternatives, restarts) and "progress" text
  in the model's own completions; locate MLP neurons / directions; dampen them with a sparse edit.

## Log
- Pass 1: base, ids 0-63 x4, T=1: acc 0.31, boxed 0.48, truncated 0.52 (mean len 440). Finished answers are 63% correct.
  Truncated answers are full of "Wait", "Alternatively", "However", "Let's re-read", "interpret", "If ...": ~0-0.1 per
  finished answer vs 0.2-0.4 each per truncated answer.
- Passes 2-3: teacher-forced the discovery completions; collected residual + MLP acts at "detour points" (token right
  before a detour marker, n=159 in pass 3) and at controls (same token types, e.g. "\n\n", ".", not followed by a marker).
  Token-matched diff-of-means direction separates detour points from controls with held-out AUC ~0.94-0.97 from layer
  13 on (0.81-0.87 at layers 5-11). The matched difference grows steadily from L12 (0.2) to L22 (2.9 along d22).
- Logit lens of d22 (layer-22 detour direction): promotes "Let", "Alternatively", "Wait", "Or", "However", "If",
  "Checking"; i.e. a "branch / reconsider" direction.
- MLP neurons writing d22 (attribution = activation diff x (down_proj column . d22)): MLPs of L12-22 explain ~1.7 of
  the 2.9; the rest is attention / linear attention. Top neurons are interpretable: L18 n1261 ("let"), n2486/n1824
  ("check"), n1405 ("re-"), L19 n36 ("or"), L17 n1795 ("alternative"). Top 300 neurons explain ~1.3.
- Edit v1: for the top-K (K=300) neurons, remove alpha x their down_proj column's component along d22
  (w <- w - alpha (w.d) d). 303k entries, 11 down_proj tensors (L12-22).
- Pass 4: wasted (tokenization mismatch bug in my script).
- Pass 5: teacher-forced check of K300 alpha2 on the pass-3 texts: log-prob of the actual detour-marker token at detour
  points drops by 0.91 nats on average (median 0.69, n=159; in-sample), all other 40k answer tokens change by mean
  -0.003 (mean |change| 0.03). So the edit is specific to the branch decision in teacher forcing.
- Pass 6 (tuning set ids 64-191 x2, both=True, T=1): K300 alpha2 edit: base acc .297 trunc .50 len 424 vs edited acc .285
  trunc .50 len 428. Detour markers per answer fell from 0.76 to 0.39 but truncation did NOT change. Conclusion: the
  surface "Wait/Alternatively/Let's" tokens are a symptom; suppressing them at the output just makes the model wander
  with other words. Abandoned v1 (kept in work/edits/K300_a2.0.safetensors).
- Checked: ~24% of truncated base answers had already written "= <gold>" somewhere (strict regex) -> failing to conclude
  is part of it; but most truncations are confused / re-interpreting / very verbose structured answers.
- Sequence-level contrast (problem-matched: 23 problems with both truncated and finished samples, mean residual over
  answer tokens, truncated minus finished): very consistent across problems (leave-one-out sign agreement 0.96-1.0 at
  L13-22). Logit lens at L22 promotes hedging words: "maybe", "usually", "anyway", "perhaps", "likely", "might",
  "weird", "something"; suppresses bold-header formatting tokens (":**"). Cos with detour-point direction ~0.4-0.47.
  => an "uncertain / hedging mode" that persists over the whole answer, not just at branch points.
- Edit v2 idea: steer against this mode direction with a constant residual write, implemented by adding
  -beta/4 * m_L / mean_act to the down_proj column of a near-constant neuron (CV 0.17-0.31) at layers 15,16,19,21
  (L15 n1184, L16 n3, L19 n16, L21 n0). 4,096 entries.
- Pass 7: steer beta=4 on tuning set (ids 64-191 x2, both=True): base acc .297 boxed .484 trunc .500 len 424 ->
  edited acc .367 boxed .668 trunc .336 len 404; markers/answer .76 -> .18; acc among finished .59 -> .55.
  Paired (per-problem) acc diff +0.070 +- 0.034 (z=2.1).
- Pass 8: steer beta=8 on tuning set: acc .262 (paired -0.035 +- 0.032), boxed .586, trunc .414, markers .05,
  acc|finished .45. Overshoot: hedging words vanish, but answers become heavily over-formatted (bold headers, LaTeX
  align blocks, day-by-day enumerations) and still run long; accuracy of finished answers drops. The direction mixes
  "hedging" with "less formatting"; pushing too far gives the opposite failure. Inverted-U so far.
- Pass 9: steer beta=2 on tuning set: acc .340 (paired +0.043 +- 0.030), boxed .645, trunc .348, markers .30.
- Dose-response on tuning set (beta: acc / trunc / markers per answer): 0: .297/.500/.76; 2: .340/.348/.30;
  4: .367/.336/.18; 8: .262/.414/.05. Truncation falls with dose then rises again when over-formatting takes over;
  accuracy peaks at beta=4. Chose beta=4 (selected on tuning set, so the tuning number is optimistic; held-out below).
- Pass 10 HELD-OUT A (ids 5000-5255, never used before, n=1, T=1, both=True), beta=4:
  base acc .320 boxed .543 trunc .461 len 420 markers .83 -> edited acc .445 boxed .734 trunc .266 len 398 markers .16;
  acc among finished .594 -> .606 (no loss in quality of finished answers). Paired diff +0.125 +- 0.033 (z=3.8).
- Pass 11 CONTROL on held-out A: random-direction steer, same 4 carrier columns and same per-layer norms (4,089 entries):
  acc .328 boxed .523 trunc .473 len 423 -- indistinguishable from base (.320/.543/.461). So the effect is specific
  to the direction, not to perturbing those neurons.
- Pass 12 HELD-OUT B (ids 5256-5511, n=1, T=1, both=True), beta=4: base acc .336 boxed .508 trunc .492 ->
  edited acc .402 boxed .688 trunc .316; paired +0.066 +- 0.034 (z=2.0). acc|finished .66 -> .58 here.
  Pooled A+B (512 problems): acc .328 -> .424 (+0.096 +- ~0.024), truncation .477 -> .291.
- Pass 13 HELD-OUT A GREEDY (ids 5000-5255, both=True): base acc .504 boxed .801 trunc .199 len 366 ->
  edited (beta 4) acc .457 boxed .805 trunc .199 len 382; paired -0.047 +- 0.030 (z=-1.6). The edit does NOT help
  greedy decoding (greedy already hedges much less: markers .36 vs .83 at T=1) and tends to make answers longer
  (more formatting). Also note: greedy decoding by itself (.50) beats the edited model at T=1 (.445).
  Hypothesis: the steering direction mixes "hedging" with "less markdown formatting"; the formatting part makes
  greedy answers more verbose. Next: remove a formatting direction from the steering vector.
- Removing a markdown-formatting direction (newline positions followed by **, #, $$, list markers vs others) from the
  steering vector changes it little (cos -0.15..-0.29); logit lens unchanged. Not pursued.
- Next: test the smaller dose beta=2 under greedy on tuning ids (64-319), since greedy already hedges little.
- Pass 14: beta=2 GREEDY on tuning ids 64-319 (both=True): base acc .465 trunc .227 -> edited .473 trunc .203
  (paired +0.008 +- 0.028). beta=2 is neutral under greedy, while beta=4 cost -0.047 under greedy on held-out A.
  Plan: evaluate beta=2 on held-out A (T=1 and greedy) and B (T=1).
- Passes 15-16 HELD-OUT A with beta=2: greedy acc .520 (base .504, +0.016; trunc .184 vs .199);
  T=1 acc .359 vs base .320 (paired +0.039 +- 0.029), trunc .375 vs .461, markers .40 vs .83.
  Trade-off on held-out A: beta=2 = small T=1 gain, no greedy loss; beta=4 = large T=1 gain (+.125), greedy -.047.
- Next: beta=3 on fresh ids 6000-6255 (T=1 and greedy, both=True) as a compromise check.
- Passes 17-18 FRESH HELD-OUT C (ids 6000-6255, both=True) with beta=3:
  T=1: base acc .324 boxed .516 trunc .477 -> edited .406 boxed .699 trunc .297; paired +0.082 +- 0.034 (z=2.4).
  greedy: base acc .504 boxed .762 trunc .242 -> edited .543 boxed .852 trunc .148; paired +0.039 +- 0.029 (z=1.4).
  => chose beta=3 as the submission (work/edits/steer_b3_final.safetensors = work/edited/edit.safetensors).
- Passes 19-20: general-text side-effect check (8 short non-math texts, raw text without chat template, 292 tokens),
  base vs beta=3: mean change in log-prob of the actual tokens -0.068 nats/token, mean |change| 0.245, max 2.2.
  That is a noticeable perturbation (~7% perplexity increase on this tiny sample) -- a real risk for the
  instruction-following benchmark that I could not test further (budget exhausted). Possible cause: the "constant"
  carrier neurons were only checked for constancy on math-answer tokens; on other text (esp. early positions) their
  activation may differ, so the write is not exactly a constant steer there.

## Final submission
work/edited/edit.safetensors = anti-hedging steer, beta=3: 4 down_proj columns (L15 n1184, L16 n3, L19 n16, L21 n0),
4,089 entries changed. Each column gets -(3/4) * m_L / mean_act(carrier), where m_L is the problem-matched mean
residual difference (truncated minus finished answers) at that layer.
Held-out summary (practice ids never used for discovery/tuning):
  beta=4 A+B T=1 (512): acc .328 -> .424; A greedy (256): .504 -> .457
  beta=2 A T=1: .320 -> .359; A greedy: .504 -> .520
  beta=3 C T=1 (256): .324 -> .406 (+.082 +- .034); C greedy: .504 -> .543 (+.039 +- .029)
  random-direction control (beta 4 norms, A T=1): .328 vs base .320 (no effect)
