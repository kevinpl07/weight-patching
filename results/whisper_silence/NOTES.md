# whisper_silence: Whisper writes text on silence, noise and music

Model: `openai/whisper-small` (HF transformers 5.20, fp32, greedy, English, no timestamps, one ≤30 s window, HF
default suppress / begin-suppress tokens; our decoder reproduces `model.generate`, checked in the original run).
Code: `tasks/whisper_silence.py` (commands), `tasks/_whisper_core.py` (decoder, thresholds, VAD, metrics),
`tasks/_whisper_prepare_data.py` (data build). Nothing here imports the old project.

## Data

Audio is not in the repo. `$WHISPER_SILENCE_DATA` (default `../data_audio`, i.e. next to `weight-patching/`) must hold
`derived/<suite>.npy|json` (16 kHz mono float32, all clips concatenated + manifest). Rebuild them from the raw downloads
with `python tasks/_whisper_prepare_data.py` (CPU, ~1 min; fixed seeds, identical output; raw sources are listed in
its docstring: LibriSpeech test-clean/dev-clean/test-other, ESC-50 master, `lewtun/music_genres_small`).

| suite | clips | role |
|---|---|---|
| `nonspeech` | 300 | held-out failure set: silence 30, noise 60, ESC-50 environmental (folds 3–5) 140, FMA instrumental music 70 |
| `padded` | 300 | held-out guard: test-clean utterances with 5–10 s of silence / noise / music on both ends |
| `wer` ("clean") | 500 | held-out guard: test-clean utterances |
| `bg` | 300 | held-out guard: test-clean with ESC-50 / music mixed under the whole utterance at SNR 0/5/10 dB |
| `other` | 300 | held-out guard: LibriSpeech test-other |
| `practice_*` | 100 + 100 + 50 + 50 | practice: non-speech (ESC-50 folds 1–2, other tracks/seeds), dev-clean speech, padded, bg |

## Metrics

- **hallucination** (failure, ↓): % of the 300 non-speech clips with a non-empty transcript after Whisper's
  `EnglishTextNormalizer` (" ." and bracketed tags like "[Music]" normalize to empty; "Music" or " you" do not).
  `hallucination_notag` also forgives bare sound tags ("Music", "Applause").
- **speech_kept** (kept, ↑): % of the 1,400 held-out guard utterances (padded + clean + bg + other) whose normalized
  transcript is *identical* to the base model's. Any change counts against it, improvements included; the WER
  columns show the direction. Base = 100 by definition.
- `metrics` also has corpus WER and % fully dropped utterances per guard suite, padded insertions per reference
  word, bg WER per SNR, and per-category hallucination.
- `guards.chat_kl`: there is no chat for an ASR model, so this slot holds the decoder's mean per-token
  KL(base ‖ arm), teacher-forced on the base transcripts of 100 held-out clean utterances. 0 for decoding-rule arms,
  null for VAD (it changes the input). `guards.mmlu200` is null.

## Arms

| arm | what | weights touched |
|---|---|---|
| base | unmodified whisper-small | 0 |
| thresholds | openai-whisper's no-speech rule with its **default** thresholds (skip if p(no-speech) > 0.6 and avg log-prob ≤ −1.0), applied deterministically to the base greedy pass (no temperature fallback) | 0 |
| thresholds_tuned | same rule, no-speech > 0.2 and no log-prob condition: the value tuned on **practice** data in the original run (`whisper_halluc/tune_thresholds.py`) | 0 |
| vad | Silero VAD in front (faster-whisper `vad_filter` defaults); no speech → empty, else the speech regions are concatenated and transcribed | 0 |
| gate | the recipe: one unit of the last decoder layer's MLP reads the non-speech direction at the first content position and writes " ." (details in `fit_gate`'s docstring) | 1,537 entries |
| random | same unit rewritten with random directions of the same norms, same bias | 1,537 entries |
| lora | LoRA SFT of the decoder (encoder frozen) on the **same 300 practice clips**: non-speech → " ." + EOT, speech/padded/bg → the base model's own transcript + EOT. patchkit's fixed recipe: r=16, α=32, dropout 0.05, q/k/v/out of self- and cross-attention + fc1/fc2 in every decoder layer, lr 2e-4, 3 epochs (min 30 steps), batch 4, seed 0. Not tuned. | 3.83 M trainable |

The gate's target " ." is the shortest output HF decoding allows (EOT is suppressed at the first step), so " ." is
"say nothing"; LoRA is trained toward the same target and therefore sees the same information (label + target).

## Held-out results (whisper-small, 2026-11-06)

Hallucination = % of 300 non-speech clips; kept = % of 1,400 guard utterances with an unchanged transcript.
WER columns are corpus WER (% of utterances fully dropped in parentheses). KL = decoder KL on clean speech.

| arm | halluc ↓ | kept ↑ | silence | noise | env | music | clean WER | bg WER | other WER | padded ins/word (dropped) | KL | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | **84.0** | 100 | 43.3 | 48.3 | 100 | 100 | 3.38 (0) | 5.77 (0) | 7.55 (0) | 1.37 (0) | 0 | – |
| thresholds (openai default) | 49.7 | 100 | 30.0 | 1.7 | 69.3 | 60.0 | 3.38 (0) | 5.77 (0) | 7.55 (0) | 1.37 (0) | 0 | 0 weights |
| thresholds_tuned (practice) | 6.7 | 98.4 | 0 | 0 | 7.9 | 12.9 | 3.77 (0.6) | 6.17 (2.0) | 8.32 (2.3) | 0.93 (2.0) | 0 | 0 weights |
| vad (Silero) | 12.3 | 89.6 | 0 | 0 | 18.6 | 15.7 | 3.33 (0) | 5.75 (0) | 7.69 (0) | 1.42 (0) | – | 0 weights |
| **gate** | **0.3** | **100** | 0 | 0 | 0 | 1.4 | 3.38 (0) | 5.77 (0) | 7.55 (0) | 1.37 (0) | **0.0** | 1,537 entries, 58 s |
| random | 84.0 | 100 | 43.3 | 48.3 | 100 | 100 | 3.38 (0) | 5.77 (0) | 7.55 (0) | 1.37 (0) | 0.0 | 1,537 entries |
| **lora** | **0.3** | 75.1 | 0 | 0 | 0 | 1.4 | 3.25 (0) | 5.78 (0) | 8.10 (0) | 0.98 (0) | 0.285 | 3.83 M params, 117 s |

Readings
- **Gate and LoRA fix the failure equally** (84% → 0.3%; the one remaining clip, `ns_music_017`, is an FMA
  "Instrumental" excerpt that both transcribe as a lyric line, " When and with you is all I've had", so it probably has
  vocals). Both arms leave the same single clip.
- **The gate is bit-exact on speech**: all 1,400 guard transcripts are identical to base and the decoder KL on clean
  speech is 0 to six decimals: the unit never fires on speech. LoRA, trained on the same 300 clips with targets that
  are the base model's *own* transcripts, changes 24.9% of speech transcripts (clean 21%, padded 19%, bg 29%,
  other 32%). The changes are mostly small rewrites ("Pauly's" → "Polly's", "Ms." → "Miss", "rest" → "rust",
  punctuation), so WER barely moves (clean −0.14, bg +0.02, other +0.55, padded insertions −0.39), but it is no
  longer the same model on speech.
- The random control does nothing (the random direction never crosses the threshold): the effect is the direction,
  not the perturbation.
- openai's default thresholds are free but only reach 49.7%. Thresholds tuned on practice reach 6.7% but drop 2–2.3%
  of real utterances entirely on padded / bg / other. VAD reaches 12.3% (ESC-50 sounds and music pass the VAD) and
  changes 10% of transcripts because it cuts the audio.
- These numbers match the original run (RESULTS_2026-11-05_practical_edits.md, Experiment 3) exactly for base,
  thresholds, tuned thresholds, VAD and the 29-line recipe (0.3%).

## Blog examples (held-out clips)

Source audio: `$WHISPER_SILENCE_DATA/derived/nonspeech.npy` at the clip's manifest offset (`nonspeech.json`); short
excerpts (first ≤8 s, 16 kHz mono WAV) are in `site/audio/<id>.wav`, licences in `site/audio/LICENSES.md`.
Transcripts are for the whole clip (raw text, before normalization; " ." normalizes to empty).

| clip | category | length | base | gate | lora | openai thresholds |
|---|---|---|---|---|---|---|
| `ns_sil_000` | digital silence | 5 s | " you" | " ." | " ." | " you" |
| `ns_noise_010` | white noise −20 dBFS | 14.4 s | " you" | " ." | " ." | "" (skipped) |
| `ns_env_110` | crackling fire (ESC-50) | 25 s | " I'm going to go get some water." | " ." | " ." | " I'm going to go get some water." |
| `ns_env_130` | rain (ESC-50) | 15 s | " The water is very clear." | " ." | " ." | " The water is very clear." |
| `ns_env_113` | dog barking (ESC-50) | 25 s | " Woof woof woof woof …" (loops to the 444-token limit) | " ." | " ." | " Woof woof woof …" |
| `pad_7729-102255-0039` | speech with 5–10 s pads (LibriSpeech, not exported) | – | " The inmates being removed at the appointed hour if you cannonballs were fired through the stone walls." | identical to base | " The inmates being removed. At the appointed hour, a few cannonballs were fired through the stone walls." | identical to base |

The last row shows what "speech_kept 75%" means for LoRA: speech transcripts drift (here for the better; elsewhere
"rest" → "rust", "Ms. Giago" → "Miss Giago"), while the gate's are byte-identical to the base model's.
