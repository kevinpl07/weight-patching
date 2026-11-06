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
