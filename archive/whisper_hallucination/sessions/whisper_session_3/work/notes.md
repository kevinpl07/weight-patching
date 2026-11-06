# Notes (session 3)

## Baseline (unedited, practice)
- halluc 0.79 overall; env 0.98, music 1.00, noise 0.35, silence 0.30. halluc_notag 0.73.
- clean WER 3.23; bg WER 6.02, bg dropped 0; padded ins 0.222% (any-ins 4%).
- Key observation: when the model does NOT hallucinate on non-speech it outputs " ." then EOT. So the model
  already has a "nothing here" answer; the failure is that at the FIRST content position " ." loses to
  prior-driven tokens (" you", " I", " Music", " I'm ...") which then often loop.
- Plan: find a direction in the decoder residual at the first content position (the <|notimestamps|> position)
  that separates non-speech from speech; add one gated MLP unit in a late decoder layer that reads it and
  writes toward the " ." unembedding (away from hallucination openers). Check the unit stays off on all
  speech positions (clean / background / padded / quiet synthetic speech).

## Evidence (one 64-clip run: 32 non-speech, 10 lowest-SNR background, 12 clean, 6 padded, 4 clean at -35 dB)
- First collection attempt (5 batches) was killed: GPU lock contention made one batch take 13 min; replaced by one balanced batch.
- At the first content position (decoder input <|notimestamps|>), the RAW top token on non-speech is " [" or " ("
  (lp ~ -0.6..-2.4), i.e. the model wants to write a bracketed sound tag ("[Music]", "(birds chirping)") which the
  normalizer would treat as empty. But " [" / " (" are in the deployed suppress list, so they are masked and the
  next-best tokens win: the language-model prior " you" / " I" / " Music" / " I'm ...". " ." (id 2411, the
  harmless empty answer) sits ~1.5-3 nats lower (and for env/music is not even in the top-6).
- nsp is high on non-speech (median 0.70; speech <= 0.15) but does NOT separate hallucinating from non-hallucinating
  non-speech clips (both ~0.70): the decoder knows there is no speech; the failure is the suppressed sound-tag route.
- Diff-of-means direction (non-speech minus speech, pos 3) in the input of each MLP (final_layer_norm output):
  layer 8: ns pos3 12.1..17.3 vs every speech position max 10.8 (speech pos3 max 1.9)
  layer 10: ns 25.5..38.7 vs speech any-pos max 19.9
  layer 11: ns 30.3..46.6 vs speech any-pos max 21.8 (pos3 max: padded 16.7, bg 6.9, clean 6.9, quiet -3.4)
  -> chose layer 11 (largest absolute gap).
## Edit v1
- Repurpose layer-11 MLP neuron 746 (least used: mean |gelu|*||fc2 col|| = 4e-5 vs median 0.04).
  fc1 row = unit direction d, bias = -(speech max 21.83 + 2.2) = -24.03; fc2 col = scale * unit u,
  u = LNgain * (E[" ."] - mean E[136 tokens seen in top-6 at pos3]) (centered). Per-unit logit effect /sigma vs
  competitors: min 1.1, median 1.8. Activation on practice ns pos3: median 14; 0 on every speech pos3.
- 1,537 entries changed.

## Results on practice
- v1 scale 8 (work/edits/L11_thr2.2_g1_s8.safetensors): halluc 0.79 -> 0.00 (env 0.98->0, music 1.0->0,
  noise 0.35->0, silence 0.30->0); clean WER 3.229 -> 3.229 (identical transcripts), bg WER 6.018 -> 6.018,
  bg dropped 0 -> 0, padded ins 0.222% / any 4% -> unchanged. Only visible change on speech: 4 padded clips
  (trailing silence/noise/music) get an extra trailing " ." which the normalizer removes. Three non-speech clips
  output " . ." or a run of " ." (empty after normalization).
- Dose check at scale 3: see below.
- Dose check at scale 3 (work/edits/L11_thr2.2_g1_s3.safetensors): metrics call launched but still queued behind
  other jobs on the shared GPU when time ran out; if it finishes it writes work/metrics_s3.json (its script then
  copies the scale-8 edit back, so the submission stays scale 8).
- The gate is the safety margin: the unit's pre-activation is <= 0 at every speech position in the probe batch
  (all decoder positions of clean, -35 dB clean, 0-10 dB background and padded clips). Threshold 24.0 vs speech
  max 21.8 (first content position max 16.7 on padded, 6.9 on background/clean); non-speech first position 30.3..46.6.
## Submission: v1 scale 8 (layer 11, neuron 746), 1,537 entries.
- Dose check result (arrived after the report): scale 3 also gives halluc 0.00 in every category, with clean WER 3.229,
  bg WER 6.018, bg dropped 0, padded ins 0.222% / 4% all unchanged. Only 1 non-speech clip outputs " . ." (3 at
  scale 8) and 3 padded clips gain a trailing " ." (4 at scale 8). So practice saturates at both strengths, and the
  threshold (gate), not the strength, decides which clips are caught. I kept scale 8 for headroom on held-out
  clips that sit just above the threshold.
