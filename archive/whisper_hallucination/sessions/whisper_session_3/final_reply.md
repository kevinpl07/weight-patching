# Session 3: agent's final reply (Opus 5.5, blind; condensed from its report to the orchestrator)

9 of 30 passes used, about 41 minutes, including about 13 minutes lost waiting for the GPU. The submission is `work/edits/L11_thr2.2_g1_s8.safetensors`. A dose check at scale 3 was still queued when time ran out.

## Mechanism (agent's claim)
- **The model wants a sound tag.** At the first content position, the raw top token on every practice non-speech clip is " [" or " (" (log-prob −0.6 to −2.4). The model is trying to write a bracketed sound tag, which the normalizer would empty out.
- **The decoder blocks it.** Those tokens are on the fixed suppress list, so the language-model prior wins (" you", " I", …) and loops follow.
- **The model knows there's no speech.** The no-speech probability is high on non-speech (median 0.70 vs ≤0.15 on speech), but it doesn't separate hallucinated from clean non-speech clips.
- **The empty answer loses.** " ." sits 1.5–3 nats below the winning token, and for environmental sounds and music it isn't even in the top 6.

A non-speech-minus-speech direction separates the classes in every late MLP input:

| Layer | Non-speech | Max on speech |
|---|---|---|
| 8 | 12.1–17.3 | 10.8 |
| 10 | 25.5–38.7 | 19.9 |
| 11 | 30.3–46.6 | 21.8 |

## Edit
Decoder layer 11, MLP neuron 746 (the least-used neuron).
- **fc1 row** = the unit-length non-speech direction.
- **Bias** = −24.03: the maximum speech value at any position (21.83) plus a margin of 2.2.
- **fc2 column** = 8 × a unit vector that raises " ." against the 136 tokens seen in the top 6 at the first position.
- **Size:** 1,537 entries. Change norms: weight 1.39, bias 22.6, fc2 8.01.

## Practice effect
| Metric | Unedited | Edited |
|---|---|---|
| Hallucination, overall | 0.79 | 0.00 |
| Silence | 0.30 | 0 |
| Noise | 0.35 | 0 |
| Environmental | 0.98 | 0 |
| Music | 1.00 | 0 |
| Clean WER | 3.229 | unchanged |
| Background WER | 6.018 | unchanged |
| Background dropped | 0 | unchanged |
| Padded insertions | 0.222% / 4% | unchanged |

There was no dose–response check.

## Predictions for the held-out scores
| Metric | Prediction |
|---|---|
| Hallucination, edited | 5–20% overall (silence 0–5%, noise 0–10%, environmental 10–25%, music 10–30%) |
| Padded insertions | unchanged |
| Clean WER | ~0 |
| Background WER | 0 to +0.3, dropped +0–1% |
| Harder-speech WER | 0 to +0.5, dropped +0–1% |
| Behind VAD | still helps |

## Late dose check (finished after the session ended)
Scale 3 (work/edits/L11_thr2.2_g1_s3.safetensors) on practice: hallucination 0.00, clean WER 3.229, background WER 6.018, background dropped 0, padded insertions 0.222%. Identical to scale 8. The submission stayed at scale 8.
