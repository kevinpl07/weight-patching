# Task: stop a speech recognizer from inventing text on non-speech audio by editing its weights

You are working on **Whisper**, a pretrained speech recognition model. When nobody is speaking it often outputs text anyway: on silence, background noise, environmental sounds or music it writes things like "you", "Thank you." or "Thanks for watching!", or it repeats a phrase until it runs out of tokens. It also adds invented words when speech is surrounded by long stretches of silence, noise or music. You will read its weights and activations and write new values into them directly. **Your edit is final.** No training happens afterwards, and the edited network is what gets scored, with the same decoding as today.

There is no training on your side: no gradients, no optimizer, and no fitting of weights by regression or by search against a loss. You may compute statistics from weights and activations (means, differences, projections, PCA/SVD, norms, per-unit activation profiles) and construct an edit from what you find. Comparing a handful of hand-chosen variants, for example to check dose–response, is fine. Automated search loops over edit parameters or over model components are not.

## The model

- **Architecture:** `openai/whisper-small` in Hugging Face transformers (`WhisperForConditionalGeneration`), float32, eval mode. An audio encoder turns 30 s of log-mel spectrogram (80 mel bins × 3,000 frames; shorter clips are zero-padded to 30 s) into 1,500 encoder frames. A transformer decoder (embedding, then layers of self-attention, cross-attention over the encoder frames and an MLP, then a final layer norm and an output projection tied to the token embedding) writes the transcript token by token. `T.info()` prints the exact sizes, special token ids and parameter names.
- **Decoding (fixed; it is what gets scored):** the decoder starts from the prompt `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` and decodes greedily until end-of-text or 444 new tokens. As in the original Whisper code, a fixed list of special tokens is suppressed at every step, and end-of-text and a bare space are suppressed at the **first** step, so every output has at least one token.
- **What counts as a hallucination:** a transcript of non-speech audio whose text, after Whisper's English text normalizer, is **not empty**. The normalizer lower-cases, removes punctuation and symbols, and deletes anything inside `[brackets]` or `(parentheses)`. So " ." or " ♪" or " [Music]" count as empty; " you", " Music" or " Thank you." do not. (A secondary rate that also ignores bare sound tags such as " Music" is reported too, but the primary rate is the one above.) `T.run` returns this normalized text as `norm_text`.

## The failure you are asked to fix (your practice data)

`T.practice()` gives you 300 clips (16 kHz mono, at most 30 s each):
- **100 non-speech clips:** silence (digital silence and near-silent noise), white and pink noise at several levels, environmental sounds (animals, machines, weather, household sounds, …) and instrumental music. Many of them get a non-empty transcript today. Look at some of them (`T.spectrogram`) and at what the model writes.
- **100 clean speech clips:** read English speech with reference transcripts.
- **50 padded speech clips:** other read-speech utterances with 5–10 s of silence, noise or music before and after, with reference transcripts. Words in the transcript that the reference does not contain are insertions.
- **50 speech-in-background clips:** other read-speech utterances with an environmental sound or music mixed under the whole utterance, at a speech-to-background ratio of 0, 5 or 10 dB, with reference transcripts. Real speech in hard conditions: it must still be transcribed, not silenced.

You may also build your own inputs (any float32 array at 16 kHz, up to 30 s) and pass them to `T.run`. You are not told what inside the network produces the made-up text; finding it is part of the task.

## How you will be scored (you will not see this data)

- **Non-speech hallucination rate:** 300 held-out non-speech clips from other recordings and sources than your practice clips (silence, noise, environmental sounds, music). Reported overall and per category; a large relative drop from the unedited model is the goal.
- **Insertions on padded speech:** 300 held-out utterances from a different speech split, each with 5–10 s of silence, noise or music before and after. Two numbers: inserted words per reference word, and the share of utterances with any inserted word. They must not rise.
- **Collateral damage on real speech**, word error rate (WER) and the share of utterances whose transcript comes out empty (fully dropped):
  - 500 held-out clean utterances: WER may rise by at most about 0.3 points absolute.
  - 300 held-out utterances with environmental sounds or music under the speech (0–10 dB speech-to-background ratio), and 300 held-out utterances of harder, less clearly read speech from a different recording set: WER may rise by at most about 1 point, and at most about 1% more utterances may come out empty.

  Deleting real words, or silencing hard speech, to get empty outputs is damage, not a fix.
- Also reported: the number of entries you changed, and your edit combined with a voice-activity-detection front end (does it still help on the non-speech that gets past the detector?).
- **Controls:**
  - the decoding-threshold fix practitioners use (no-speech probability, average log-probability and compression-ratio thresholds), tuned on the same practice data
  - a voice-activity-detection front end, and the two combined
  - random edits that change the same entries by the same amount
  - simple scripted edits built from the same practice data

  The question is whether careful reading and editing of the weights fixes the failure without training, without data beyond the practice set, and without hurting transcription of real speech.

## Tools

The helpers are in `whisper_halluc/agent_tools.py` (read it, but don't modify it). Run every script from this folder:

```
cd <project>/runs/whisper/session_2
PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 AGENT_SESSION=. ../../../.venv/Scripts/python.exe your_script.py
```

Start each script with:

```
import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
```

- **Free:**
  - `T.info()`: model summary, decoding rules, special token ids, budgets, the practice set
  - `T.base_weights()`: all weights as float32 CPU tensors
  - `T.practice()`: the practice clips (audio, kind, category, reference text)
  - `T.spectrogram(clips, "work/name.png")`: saves the log-mel inputs of up to 16 clips as one image; open it with your file-reading tool to look at it
  - `T.tokenizer()`: the Whisper tokenizer
  - `T.current_edit()`, `T.save_edit({name: full_new_tensor}, note=...)`, `T.budget()`
- **Counted forward passes (budget 30 passes):**
  - `T.run(clips, edited=..., ...)` (1 pass, at most 64 clips): the transcript, generated tokens, the no-speech probability, top-k log-probs at every position, and on request the full logits, the encoder output, decoder self- and cross-attention maps, the decoder residual stream after every layer, and the output of any named decoder module. `forced=` teacher-forces token sequences you choose instead of generating. Read its docstring.
  - `T.metrics(edited=...)` (2 passes, about 1–2 minutes): practice diagnostics: hallucination rate overall and per category, WER on the clean clips, WER and dropped share on the speech-in-background clips, insertions on the padded clips, and every practice transcript.
- The GPU is shared with other jobs; every call waits for its turn, so a call can take a few minutes. Request only the internals you need, and cache what you reuse (for example with `torch.save` into `work/`) instead of recomputing it.

## Rules

- **Edit budget:** at most **1,535,808** weight entries may differ from the original, which is 1% of the decoder's entries. Only decoder tensors (`model.decoder.*`, including the tied token embedding) may be edited; the encoder is read-only. Smaller is fine.
- **No training:** no gradients or autograd, no optimizer, and no fitting weights by regression or search against a loss (see above).
- **Show your evidence.** Explain:
  - what produces the made-up text and where it lives in the network
  - why your edit removes it, and why it should leave real speech alone
  - what the edit does right away on practice: hallucination rate per category, clean WER, speech-in-background WER and dropped share, padded insertions

  Check dose–response. Remember the practice set is small: 100 non-speech clips. Don't overfit it, and think about quiet, short, noisy or hard-to-hear speech that your edit might now drop.
- **Stay in your lane.** Work only in this folder, and read only `whisper_halluc/agent_tools.py` and `whisper_halluc/common.py` outside it. Don't open other `runs/` folders, the `data_audio/` folder (use `T.practice()`), other scripts in `whisper_halluc/` or anywhere else in the repository, and don't use the internet.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. Keep copies of promising edits. The current edit lives in `work/edited/edit.safetensors`. Don't print huge tensors.
- **Time:** you have about **40 minutes** and about 70 tool calls. Make sure a sensible edit is saved well before the end.

## Submission

Whatever is in `work/edited/edit.safetensors` at the end is your submission. It may be empty if you find no edit you trust. Your final reply should cover:

1. how you approached the problem, step by step
2. what produces the made-up text, with the evidence
3. the edit: which tensors, how many entries, and the norm of the change
4. its measured effect on practice: hallucination rate overall and per category, clean WER, speech-in-background WER and dropped share, padded insertions
5. your prediction for the held-out scores, with how confident you are:
   - non-speech hallucination rate, unedited and edited, overall and for silence / noise / environmental / music
   - padded-speech insertion rate and share of utterances with any insertion, unedited and edited
   - the change in WER on clean speech, on speech with background sound, and on harder speech, and the change in the share of fully dropped utterances
   - whether the edit still helps behind a voice-activity-detection front end
