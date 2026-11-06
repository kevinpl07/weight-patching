# Experiment 3: Whisper hallucinating on silence, noise and music

Code for section 3 of `PLAN_2026-11-04_practical_edits.md`. Model: `openai/whisper-small` (HF transformers 5.20),
English transcription, greedy, single ≤30 s window. Every script takes `--model`, so
`openai/whisper-large-v3-turbo` can be swapped in (run `prepare_data.py` once; everything else is model-agnostic).

All commands run from the repo root with
`PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 .venv/Scripts/python ...`. Every CUDA use goes through
`gpu_lock.py` in chunks of a few minutes. GPU minutes below are measured on the RTX 4070 (fp32) and exclude lock waits.

## Files

| file | what |
|---|---|
| `common.py` | paths, suites, model/edit loading, **`apply_edit`** (shared helper), greedy decoder (= HF default `generate`, verified), openai statistics (no-speech prob, avg log-prob, compression ratio), threshold rule, Silero VAD, metrics |
| `prepare_data.py` | builds `data_audio/derived/*.npy|json` from `data_audio/raw/` (CPU, 30 s) |
| `eval.py` | scores one arm (model + edit + thresholds + VAD), resumable, every transcript saved |
| `tune_thresholds.py` | threshold grid on practice data only → `runs/whisper/thresholds*.json` |
| `head_scan.py` | single-head ablation scan + top-k → `runs/whisper/head_scan/edit.safetensors` |
| `agent_tools.py`, `session.py`, `prompt_template.md` | blind agent sessions (as in `vision_spurious/`) |
| `ct2_check.py` | converts an edited checkpoint to CTranslate2 and checks transcripts match HF (CPU) |
| `summarize.py` | the plan's pass-bar table → `runs/whisper/summary.md` |
| `smoke_decode.py`, `smoke_agent_tools.py` | smoke tests (already run; outputs in `runs/whisper/smoke/`) |

## Data (`data_audio/`)

Raw downloads (untrusted, only read as audio): `raw/librispeech/` (openslr test-clean, dev-clean, test-other),
`raw/esc50/` (GitHub ESC-50 master), `raw/music/lewtun_music_genres_small/` (HF parquet, FMA 30 s clips).

| suite | clips | content |
|---|---|---|
| `nonspeech` (held-out) | 300 | silence 30 (6 digital zeros + 24 near-silent white/pink at −80/−70 dBFS), noise 60 (white/pink at −50/−35/−20 dBFS), environmental 140 (ESC-50 folds 3–5, 1–6 same-class recordings concatenated), music 70 (FMA "Instrumental" + "Chiptune / Glitch" excerpts); 5–30 s |
| `padded` (held-out) | 300 | test-clean utterances (2–18 s), 5–10 s pads both ends: 100 digital silence, 100 white/pink noise (−45…−35 dBFS), 100 music (−35…−28 dBFS, held-out tracks) |
| `wer` (held-out) | 500 | other test-clean utterances (≤29.5 s) |
| `practice_nonspeech` | 100 | silence 10, noise 20 (other seeds), environmental 45 (ESC-50 folds 1–2), music 25 (other tracks) |
| `practice_speech` | 100 | dev-clean utterances |
| `bg` (held-out hard guard) | 300 | test-clean utterances disjoint from `padded`/`wer`, mixed over their whole length with held-out ESC-50 (folds 3–5, one class, recordings concatenated) or held-out music, at SNR 0/5/10 dB (50 per env/music × SNR cell) |
| `other` (held-out hard guard) | 300 | LibriSpeech test-other utterances (≤29.5 s) |
| `practice_padded` | 50 | other dev-clean utterances with practice pads (added; see deviations) |
| `practice_bg` | 50 | other dev-clean utterances mixed with practice ESC-50 (folds 1–2) / practice music at SNR 0/5/10 dB |

ESC-50 classes `crying_baby` and `laughing` are excluded (speech-like vocalisations); `coughing`, `sneezing`,
`snoring`, `breathing` are kept and flagged `human_vocal` in the manifest. Digital silence gives the same Whisper input
at every length (Whisper zero-pads to 30 s), so the 6 held-out digital-silence clips are one input counted 6 times.
FMA track 10150 ("Instrumental", but Whisper transcribed sung lyrics; was practice `pns_music_010`) is excluded.

## Metrics

Text is normalized with Whisper's `EnglishTextNormalizer` (removes punctuation, symbols, `[...]`, `(...)`).
- hallucination rate (primary) = share of non-empty normalized transcripts on non-speech, overall and per category;
  secondary `rate_notag` also ignores outputs that are only a sound tag ("Music", "Applause", "upbeat music";
  ≤4 words from `common.SOUND_TAG_WORDS`; bracketed tags are already empty after normalization)
- padded speech: inserted words per reference word (jiwer alignment) and share of utterances with any insertion
- WER (jiwer, corpus level) on `wer`, `bg` and `other`, plus `dropped` = share of utterances whose transcript is
  empty after normalization (fully dropped), with `bg` broken down by background type and SNR

Pass bar (plan section 3 as revised by the orchestrator, implemented in `summarize.py`): non-speech hallucination
−50% relative (strict: ≤⅓ of base); padded insertions ≤ base + 0.2 points (replaces "halves"); clean WER ≤ +0.3;
`bg` and `other` WER ≤ +1.0 with fully dropped share ≤ +1 point; random edit < 10% relative improvement.

Note: the deployed decoding (HF and openai) suppresses end-of-text at the first step, so the shortest output is one
token; " ." or " ♪" count as empty, " you" or " Music" do not.

## Command sequence

```
# 0. (done) data + smoke tests + viability check + threshold tuning
python whisper_halluc/prepare_data.py                       # CPU, 40 s (--only other: rewrite just that suite)
python whisper_halluc/smoke_decode.py                       # ~1 GPU min: our decoder == model.generate (36/36)
python whisper_halluc/tune_thresholds.py                    # ~1.5 GPU min (practice) -> thresholds.json, thresholds_strict.json
python whisper_halluc/tune_thresholds.py --vad              # ~1 GPU min -> thresholds_vad.json, thresholds_vad_strict.json

# 1. script arm: head-ablation scan (practice only)          ~55 GPU min (self only: ~28) in ~3-min lock holds; resumable
python whisper_halluc/head_scan.py                          # self + cross heads; --kinds self halves it

# 2. held-out arms (each ~4-5 GPU min unless noted)
python whisper_halluc/eval.py --arm base
python whisper_halluc/eval.py --arm thresholds        --thresholds runs/whisper/thresholds.json        --from-arm base   # no GPU
python whisper_halluc/eval.py --arm thresholds_strict --thresholds runs/whisper/thresholds_strict.json --from-arm base   # no GPU
python whisper_halluc/eval.py --arm openai_thresholds --thresholds openai                              --from-arm base   # no GPU
python whisper_halluc/eval.py --arm vad --vad                                  # VAD is CPU (cached); fewer clips reach Whisper
python whisper_halluc/eval.py --arm vad_thresholds        --vad --thresholds runs/whisper/thresholds_vad.json        --from-arm vad  # no GPU
python whisper_halluc/eval.py --arm vad_thresholds_strict --vad --thresholds runs/whisper/thresholds_vad_strict.json --from-arm vad  # no GPU
python whisper_halluc/eval.py --arm head_scan     --edit runs/whisper/head_scan/edit.safetensors
python whisper_halluc/eval.py --arm head_scan_vad --edit runs/whisper/head_scan/edit.safetensors --vad
python whisper_halluc/session.py random --edit runs/whisper/head_scan/edit.safetensors --out runs/whisper/head_scan/random0_count_edit.safetensors --mode count
python whisper_halluc/eval.py --arm head_scan_random --edit runs/whisper/head_scan/random0_count_edit.safetensors

# 3. blind sessions i = 1..3 (about 40 min each; the agent's tool calls take the GPU lock themselves)
python whisper_halluc/session.py prepare --session runs/whisper/session_1     # -> PROMPT.md; give it to the agent
#    the agent runs from runs/whisper/session_1 with AGENT_SESSION=. (see PROMPT.md)
python whisper_halluc/session.py apply  --session runs/whisper/session_1      # -> submitted_edit.safetensors
python whisper_halluc/session.py random --session runs/whisper/session_1 --seed 0 --mode same
python whisper_halluc/session.py random --session runs/whisper/session_1 --seed 0 --mode count
python whisper_halluc/eval.py --arm agent1       --edit runs/whisper/session_1/submitted_edit.safetensors
python whisper_halluc/eval.py --arm agent1_vad   --edit runs/whisper/session_1/submitted_edit.safetensors --vad
python whisper_halluc/eval.py --arm random1_same  --edit runs/whisper/session_1/random0_same_edit.safetensors
python whisper_halluc/eval.py --arm random1_count --edit runs/whisper/session_1/random0_count_edit.safetensors
python whisper_halluc/ct2_check.py --edit runs/whisper/session_1/submitted_edit.safetensors --name session_1   # CPU, ~1-2 min

# 4. table
python whisper_halluc/summarize.py                          # -> runs/whisper/summary.md / summary.json
```

Useful options: `eval.py --suites nonspeech --limit 50` for quick looks (summarize flags limited arms),
`--fresh` to discard saved transcripts, `--batch` (default 32; ~3.5 GB of cross-attention cache in fp32),
`--chunk` clips per lock hold (default 150, about 1–3 min).

## GPU time (whisper-small, fp32, measured on practice data)

| step | GPU min |
|---|---|
| one full held-out arm (base): nonspeech 300 + padded 300 + wer 500 + bg 300 + other 300 | 3.7 measured (nonspeech 85 s: hallucination loops run to 444 tokens; each speech suite 30–50 s) |
| threshold arms with `--from-arm` | 0 |
| VAD arms | ≤ 4.5 |
| head scan, self + cross (288 heads, ~11 s each incl. practice_bg) + top-k (10 × ~17 s) | ~55 (self only ~28); ~3.5 h wall under today's lock contention |
| one agent session | ≤ 30 passes × ≤ 1 min (≈ 10–20 min of GPU inside a 40-min session) |
| all planned arms (base, 4 threshold variants, vad, vad+thr, head scan ×3, 3 × (agent, agent+vad, 2 random)) | ≈ 65, plus the scan (45) and the sessions (≈ 45) |

## Decisions and deviations (see the final report too)

- **Thresholds** are applied deterministically to one greedy pass (openai rule: skip if no-speech prob > T_ns and
  avg log-prob ≤ T_lp), no temperature fallback. HF 5.20 *does* honour `no_speech_threshold`, `logprob_threshold`
  and `compression_ratio_threshold` for short-form input, but via stochastic temperature fallback (checked in
  `runs/whisper/smoke/decode_check.json`: on a music clip fallback turned "Music" into a made-up sentence).
  `condition_on_previous_text` is moot for single-window clips. `thresholds.json` = openai rule only (faithful arm 2);
  `thresholds_strict.json` adds "drop if avg log-prob < x / compression ratio > y" (a stronger, less standard variant).
  Selection mirrors the pass bar on practice: lowest practice hallucination with clean WER ≤ +0.3, practice_bg WER
  ≤ +1.0 and no more than +1 point of fully dropped practice_bg utterances, padded insertions ≤ +0.2.
- **VAD**: Silero (pip `silero-vad` 6.2.3, CPU, JIT model) with faster-whisper's `vad_filter` defaults (threshold 0.5,
  min silence 2 s, pad 400 ms); no speech → empty transcript; otherwise the speech regions are **concatenated** and
  transcribed (faster-whisper behaviour).
- **Agent scope**: decoder tensors only (incl. the tied embedding/output projection); encoder read-only
  (`AGENT_ALLOW_ENCODER=1` to change). Budget 1% of decoder entries = 1,535,808 (whisper-small); 30 passes, where
  `run` (≤64 clips) costs 1 and `metrics` (all 300 practice clips) costs 2.
- **Practice padded set** (50 dev-clean utterances) added so the agent and the scripts can see insertions.
- **Head scan** decodes non-speech with a 24-token cap (only emptiness matters) and reuses encoder outputs; speech
  guard = practice clean + practice_bg; the top-k curve and the k choice use full decoding and the same practice
  criteria as the threshold tuning.

## Viability check and practice findings (base whisper-small)

From `runs/whisper/tune/practice_base/` and `runs/whisper/smoke/` (practice data only):

| | overall | silence | noise | environmental | music | clean WER | padded ins/word | padded any-ins |
|---|---|---|---|---|---|---|---|---|
| base | 79% | 30% | 35% | 98% | 100% | 3.23 | 0.22% | 4% |
| openai default thresholds (0.6 / −1.0) | 45% | 30% | 0% | 60% | 60% | 3.23 | 0.22% | 4% |
| tuned thresholds (`thresholds.json`: ns 0.2, lp −0.2) | 15% | 0% | 0% | 27% | 12% | 3.23 | 0.22% | 4% |
| tuned strict (`thresholds_strict.json`) | 0% | 0% | 0% | 0% | 0% | 3.23 | 0.22% | 4% |
| Silero VAD | 14% | 0% | 0% | 18% | 24% | 3.57 | 0.22% | 4% |
| VAD + thresholds (ns 0.3) | 0% | 0% | 0% | 0% | 0% | 3.57 | – | – |

Typical outputs: " you" / " I" / " Music" (one token), " ." (counts as empty), and 444-token loops on periodic
sounds ("Beep Beep …", "dog barking dog barking …"). Held-out VAD (model-free, cached): 260 of 300 non-speech clips
have no detected speech; all speech clips do.

Practice note (before the hard guards): base insertions on practice padded speech (0.22%) were no higher than on
clean speech, and louder pads didn't raise them (`runs/whisper/smoke/pad_level_probe.json`), so the insertion
criterion became "doesn't rise". The practice table above predates the re-tune with `practice_bg`; the current
practice choices are in `runs/whisper/thresholds*.json` (openai rule: no_speech 0.2, no log-prob condition;
practice 79% -> 4% with no practice WER or drop cost; VAD + thresholds: no_speech 0.3).

## Held-out results (2026-11-05; `runs/whisper/eval/*/summary.json`, table via `summarize.py`)

Hallucination in % of 300 non-speech clips; "drop" = % of utterances with an empty transcript.

| arm | halluc | no-tag | sil | noise | env | music | padded ins/word % | any-ins % | clean WER (drop) | bg WER (drop) | other WER (drop) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 84.0 | 79.7 | 43.3 | 48.3 | 100 | 100 | 1.37 | 7.3 | 3.38 (0.0) | 5.77 (0.0) | 7.55 (0.0) |
| openai default thresholds (0.6/−1.0) | 49.7 | 45.7 | 30.0 | 1.7 | 69.3 | 60.0 | 1.37 | 7.3 | 3.38 (0.0) | 5.77 (0.0) | 7.55 (0.0) |
| thresholds (arm 2, openai rule tuned) | 6.7 | 6.0 | 0 | 0 | 7.9 | 12.9 | 0.93 | 6.3 | 3.77 (0.6) | 6.17 (2.0) | 8.31 (2.3) |
| thresholds_strict (arm 2b) | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0.93 | 6.3 | 3.77 (0.6) | 6.18 (2.3) | 8.30 (2.7) |
| vad | 12.3 | 12.3 | 0 | 0 | 18.6 | 15.7 | 1.42 | 8.0 | 3.33 (0.0) | 5.75 (0.0) | 7.69 (0.0) |
| vad + thresholds | 0.3 | 0.3 | 0 | 0 | 0 | 1.4 | 1.42 | 8.0 | 3.34 (0.2) | 5.78 (0.3) | 7.81 (0.7) |
| head_scan (k=1: self L11 H7) | 79.7 | 73.0 | 30.0 | 33.3 | 100 | 100 | 1.22 | 7.7 | 3.40 (0.0) | 5.96 (0.0) | 7.55 (0.0) |

Readings: the tuned thresholds now pay on the hard guards (clean +0.39 fails +0.3; bg and other drop 2.0–2.7% of
utterances, failing "≤ +1 point"), so arm 2 / 2b fail the pass bar on collateral damage. VAD + thresholds passes
everything (0.3% hallucination, drops ≤ 0.7%). The head-ablation scan barely helps: no single decoder head
(self or cross) lowers practice hallucination by more than 2 points; top-k ablation reaches 71% at k ≥ 5 but
raises practice padded insertions beyond +0.2, so k = 1 is chosen. Base `bg` WER by SNR: 9.8 (0 dB), 4.3 (5 dB),
3.3 (10 dB).
