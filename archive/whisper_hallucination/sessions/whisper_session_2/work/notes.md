# Notes (session 2)

## Baseline (unedited, practice)
halluc 0.79 overall; silence 0.30, noise 0.35, environmental 0.98, music 1.00 (notag 0.73).
clean WER 3.23; bg WER 6.02, bg dropped 0.0; padded ins 0.222% , any-ins 0.04.
Transcripts: non-hallucinating silence/noise clips output " ." (token 2411) then EOT, i.e. the model already has an
"empty" path. Hallucinating ones choose a content token at the first step: " you" (most common), " I", " Music",
or start a long loop ("Beep Beep...", "I'm not sure ..."). The decisive moment is the first content token
(position 3, after <|notimestamps|>), where EOT is suppressed by the decoding rules.

## Plan
Find a direction in the decoder residual stream at the first content position that separates non-speech from speech
(mean difference, checked on padded/background speech too). Build one gated unit in the last decoder MLP that reads
that direction and, when it fires, writes a direction raising the logits of " ." and EOT (through the final LN).

## Finding: a "no-speech" direction at the first content position (layer 11 MLP input)
Collected the input to the last decoder MLP (layers.11.final_layer_norm output) at every position, base model.
d = mean(non-speech, pos 3) - mean(speech, pos 3), unit norm. Projections onto d:
- non-speech at pos 3 (the position that picks the first content token): min 26.7, p10 29.8, median 38.1
- clean speech at pos 3: max 9.3, median -4.2
- clean speech at every position that predicts a real word: max 9.3, p99.9 6.0, median -8.1
- clean speech at the end-of-text position: up to 20.9 (fine: EOT/" ." there is harmless)
So the decoder already "knows" there is no speech (that's also what <|nospeech|> reads), but at the first content
step EOT is forbidden, and the usual fall-back token " ." loses to " you"/" I"/content tokens, after which the LM
prior writes plausible text or loops. The made-up text is the language-model prior taking over once the first token
is a word; the gate to stop it lives in the late decoder (no-speech direction feeding the unembedding).
Logit gap: " ." is outside the top-5 at pos 3 for 77/79 hallucinating clips.

## Edit v1 (t18_g2.5)
Repurpose layer-11 MLP unit 304 (least used: mean |GELU| x ||fc2 col|| = 2e-5 vs median 0.042).
fc1.weight[304] = 0.5*d, fc1.bias[304] = -0.5*18 (fires only when proj > 18; GELU leak at speech max 9.3 is ~1e-4).
fc2.weight[:,304] = 2.5*v, v = unit((E[" ."]-Ebar) + 0.5(E[eot]-Ebar)) / final_LN_gain, Ebar = mean embedding of the
34 tokens in the non-speech top-5 at pos 3. Per unit, this raises " ." by ~1.36 and EOT by ~1.58 logit-units vs
~0 for the competitors (" you" 0.08, " I" 0.28). 1,537 entries changed.

### v1 result on practice (edits/edit_t18.0_g2.5.safetensors; direction from non-speech vs clean speech only)
halluc 0.79 -> 0.00 (silence 0.30->0, noise 0.35->0, env 0.98->0, music 1.00->0); clean WER 3.229 -> 3.229;
bg WER 6.018 -> 6.018, bg dropped 0 -> 0; padded ins 0.222% -> 0.222%, any-ins 0.04 -> 0.04.
Every non-speech output became " ." (normalizes to empty). No clean/background transcript changed.
3 padded clips gained a leading " ." before the full correct transcript (harmless to the metric, but a sign the
unit fires weakly on long-padded speech).

### Padded/background projections (collected after v1)
With d recomputed on all groups: non-speech pos3 min 27.0, p10 30.4; padded speech pos3/word-position max 20.7;
background speech max 8.1; clean max ~9. theta=18 sits below the padded max, so v2 raises theta to 22.5
(midway between padded max 20.7 and non-speech min 27.0) to protect long-padded / quiet speech.

### v2 result on practice (SUBMITTED: edited/edit.safetensors = edits/edit_t22.5_g2.5.safetensors)
Unit 746 of layer 11 (least-used under all collected data), theta 22.5, gain 2.5, k 0.5. 1,537 entries changed
(fc1.weight row 746: 768, fc1.bias[746]: 1, fc2.weight col 746: 768). Delta norms: fc1.w 1.10, fc1.b 9.84, fc2.w 2.55.
halluc 0.00 overall and in every category; clean WER 3.229 (unchanged); bg WER 6.018 (unchanged), dropped 0;
padded ins 0.222%, any-ins 0.04 (unchanged); padded WER 3.226 (unchanged). No speech transcript changed at all;
the " ." prefixes seen with v1 on 3 padded clips are gone. All 100 non-speech clips now output " .".
Dose-response done only on the threshold (18 vs 22.5: same non-speech effect, 22.5 removes the padded side effect).
Not done (out of time, shared GPU queue ~8 min per call): a lower-gain check and a test on synthetic quiet speech.
Margin: unit activation at the weakest non-speech clip is 0.5*(27.0-22.5)=2.25 vs ~10 at the median.
