# Notes: Whisper non-speech hallucination weight edit

## Plan
1. Baseline practice metrics + transcripts (T.metrics, base).
2. Capture last decoder layer internals (fc1 input/pre-act, layer output, final LN output) on all 300 practice clips.
3. Find a "no speech / nothing to transcribe" direction in the last layer's MLP input; check how well it separates
   hallucinated positions (non-speech clips) from real speech positions (clean, background, padded speech).
4. Edit idea: repurpose one weak neuron of layer-11 MLP into a gated detector (fc1 row = detector direction, bias =
   threshold, GELU gating so it is ~0 on speech) whose fc2 column writes "end-of-text" (and, for the very first step
   where EOT is suppressed, a punctuation-only token) into the residual.

## Baseline (unedited) practice metrics
halluc 0.79 (silence 0.30, noise 0.35, env 0.978, music 1.00); notag 0.73; clean WER 3.229; bg WER 6.018, bg dropped 0;
padded ins 0.222%/ref word, any-ins 0.04.
When the model does NOT hallucinate it writes " ." (token 2411) then EOT. Hallucinations start with " you", " I",
" You", " I'm", " Music", ... and environmental/music often loop until 444 tokens.
=> The "empty" behaviour already exists: " ." followed by EOT. The fix is to make " ." win the first step (EOT is
suppressed there) and EOT win afterwards, but only when the decoder has no speech to read.

## Forced " ." on all 100 non-speech clips (base weights, 2 passes)
After teacher-forcing " ." as the first content token, end-of-text is the top-1 next token on 100/100 non-speech clips
(EOT log-prob median -0.22). So the made-up text is decided entirely at the FIRST content step (position 3, the
<|notimestamps|> query position): EOT is suppressed there by the decoding rules, and the model's runner-up choices
are " you"/" I"/" Music"... instead of " .". Fixing that one decision fixes the whole transcript.
(Note: T.run with full 444 tokens crashes in the teacher-forced pass (449 > 448 positions); use max_new_tokens=443.)

## Where the decision lives / detector
Captured layer-11 (last decoder layer) MLP input at position 3 (the query that picks the first content token).
Because the prompt is identical for every clip, any difference at position 3 comes only from the audio via
cross-attention. Mean difference (non-speech minus speech, position 3) = direction d1. Projection x3.d1u:
non-speech 27.3..46 (silence 27-35, noise 31-40, env 28-43, music 34-46); clean speech -11..10.9; background speech
-7.5..8.6; padded speech (5-10 s leading silence/noise/music) -5.8..23.2. So there is a clean gap on practice
(27.3 vs 23.2), padded speech being closest.
Margin needed at position 3: logit(" .") minus best competing allowed token is -5.1..+0.7 on non-speech
(silence/noise often already pick " .").
A position-gate direction (pos3 mean minus later-position mean) did NOT separate positions (later positions reach 32);
dropped it.

## Edit
Layer 11 MLP neuron 1814 has zero mean |activation| on all practice positions (dead neuron) -> repurposed it:
fc1 row = 3*d1u, fc1 bias = -3*25.5 (fires only when x.d1u > ~25.5; GELU ~0 below), fc2 column = 8 * unit vector
gamma*(2E[" ."] - E["."] - mean E) (centered), which raises " ." over "." and the rest.
Offline exact simulation (layer 11 is last layer, so effect at a position = final LN + unembedding of h11 + delta):
 - variant boosting gamma*(E[" ."]-mean E) mostly promoted "." (13) -> did not work (0/100).
 - variant boosting gamma*(E[" ."]-E["."]) promoted other junk tokens -> 0/100.
 - chosen variant B: 100/100 non-speech pick " ." at step 1 (then EOT is already top-1 after " ." on 100/100).
   On the 200 speech clips (clean, background, padded) no token changes except 3 padded clips where the EOT at the
   very end (trailing silence) becomes " ." (then EOT) -> no word change.
 - gain 8 and 16 give the same practice outcome (dose: flat above 8); gate a=3 with t=25.5.

## Measured on practice (T.metrics, edited; saved edit = work/edited/edit.safetensors, copy work/edit_t25.5_g8.0.safetensors)
halluc 0.79 -> 0.00 (silence 0.30->0, noise 0.35->0, env 0.978->0, music 1.00->0); clean WER 3.229 -> 3.229;
bg WER 6.018 -> 6.018, bg dropped 0 -> 0; padded ins 0.222 -> 0.222, any-ins 0.04 -> 0.04 (unchanged, matches the
offline simulation exactly). Edit: 1,537 entries (layer-11 fc1 row+bias and fc2 column of neuron 1814).
Risks: threshold 25.5 sits in a narrow practice gap (non-speech min 27.3, padded-speech max 23.2). Held-out padded
utterances with long leading silence could be fully dropped; quiet near-silence clips could fall below threshold.
The edit does not touch insertions that happen mid-transcript (the 2 practice padded clips with insertions).
