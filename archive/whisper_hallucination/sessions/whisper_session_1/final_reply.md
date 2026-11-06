# Session 1: agent's final reply (Opus 5.5, blind; reproduced from its report to the orchestrator)

13 of 30 passes used, about 31 minutes. Edit: 1,537 entries in decoder layer 11's MLP.

## Approach
1. **Baseline.** When the model handles non-speech correctly, it writes " ." (token 2411) and then end-of-text. When it hallucinates, it writes " you", " I", " Music" and so on, and environmental sounds and music often loop until the token limit.
2. **Forced " ." test.** Forcing " ." as the first content token made end-of-text the top next token on 100/100 non-speech clips. The hallucination is therefore decided at the first content step (position 3). End-of-text is suppressed there, and " ." loses to common opening words by up to 5.1 logits.
3. **Detector direction d1.** The input to the layer-11 MLP at position 3, mean of non-speech minus mean of speech, separates the two cleanly:

   | Practice clips | Score range |
   |---|---|
   | Non-speech | 27.3 to 46 |
   | Clean speech | −11 to 10.9 |
   | Speech with background | −7.5 to 8.6 |
   | Padded speech | −5.8 to 23.2 |

4. **Variants offline.** Layer 11's MLP output at a position feeds only that position's logits, so each variant's effect can be computed exactly on CPU from cached activations. It compared 6 hand-picked variants and 2 output vectors, then confirmed the result with `T.metrics`.

## Mechanism (agent's claim)
Whisper already detects non-speech. Its natural "nothing here" output is " ." followed by end-of-text, but the first-step end-of-text suppression and the opening-word prior override it.

## Edit
- **Neuron.** It repurposed dead neuron 1814 in layer 11's MLP:
  - The `fc1` row becomes 3·d̂1 (change norm 3.25).
  - The bias becomes −3·25.5 (change norm 75.2), so the neuron fires only when x·d̂1 > 25.5.
  - The `fc2` column becomes 8 × the unit vector of the centered γ∘(2E[" ."] − E["."] − mean E) (change norm 8.03).
- **Dose–response:**
  - Gain 8 and gain 16 give the same practice result.
  - Threshold 24 fires on padded speech at the first step.
  - Threshold 27 fires at the end of some speech.

## Practice effect
| Metric | Unedited | Edited |
|---|---|---|
| Hallucination, overall | 0.79 | 0.00 |
| Silence | 0.30 | 0 |
| Noise | 0.35 | 0 |
| Environmental | 0.978 | 0 |
| Music | 1.00 | 0 |
| Clean WER | 3.229 | 3.229 |
| Background WER | 6.018 | 6.018 |
| Background dropped | 0 | 0 |
| Padded insertions | 0.222% | 0.222% |

## Predictions for the held-out scores
| Metric | Prediction |
|---|---|
| Hallucination, unedited | ~0.75 |
| Hallucination, edited | 0.10–0.25 overall (silence 0.05–0.2, noise 0.05–0.2, environmental 0.15–0.3, music 0.1–0.25) |
| Padded insertions | unchanged, 0.2–0.4% |
| Clean WER | ±0.1 |
| Background WER | ±0.3, dropped ~0 |
| Harder-speech WER | 0 to +0.3 |
| Dropped padded utterances | a few of 300 at most |
| Behind VAD | still helps on environmental sounds and music |

Confidence: moderate. The threshold is only 1.8 below the lowest practice non-speech score.
