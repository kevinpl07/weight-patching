# Session 2: agent's final reply (Opus 5.5, blind; condensed from its report to the orchestrator)

12 of 30 passes used. Two of those were lost: one crashed on a 449-position overflow, and one was killed by a `| head` pipe. The session ran past 40 minutes because the GPU queue held each call for 4–8 minutes. The submission is `work/edits/edit_t22.5_g2.5.safetensors`.

## Mechanism (agent's claim)
The layer-11 MLP input at position 3, the first content step, holds a strong "no speech" direction d, the difference of means between non-speech and speech. Projections onto d:

| Clips | Positions | Projection |
|---|---|---|
| Non-speech | first content step | min 27.0, p10 30.4, median 37.8 |
| Clean speech | all word-predicting positions | max 9.3 |
| Speech with background | all word-predicting positions | max 8.1 |
| Long-padded speech | all word-predicting positions | max 20.7 |

End-of-text is forbidden at the first step, and the empty fallback " ." is outside the top 5 on 77 of the 79 hallucinating clips. The failure therefore sits in the last stretch from that representation to the output, not in the encoder.

## Edit
The edit repurposes unit 746 in `layers.11`, the least-used unit.
- **Read:** fc1 row = 0.5·d, bias = −0.5·22.5. The unit fires when the projection exceeds 22.5, midway between the padded-speech maximum and the non-speech minimum.
- **Write:** fc2 column = 2.5·v, where v = (E[" ."] − Ē) + 0.5·(E[eot] − Ē), divided by the final layer-norm gain and normalized. Ē is the mean embedding of the 34 top-5 tokens.
- **Size:** 1,537 entries. Change norms: fc1 weight 1.10, bias 9.84, fc2 2.55.

## Practice effect
| Metric | Unedited | Edited |
|---|---|---|
| Hallucination, overall | 0.79 | 0.00 (all clips output " .") |
| Clean WER | 3.229 | unchanged |
| Background WER | 6.018 | unchanged |
| Background dropped | 0 | unchanged |
| Padded insertions | 0.222% / 4% | unchanged |

Dose: at threshold 18 the result was the same, but 3 padded clips gained a leading " .". The submission uses 22.5. The gain was not varied.

## Predictions for the held-out scores
| Metric | Prediction |
|---|---|
| Hallucination, edited | 0.03–0.12 overall (silence and noise ~0, environmental and music 0.05–0.15) |
| Padded insertions | equal or lower |
| Clean WER | ±0.05 |
| Background WER | +0–0.3, dropped +0–1% |
| Harder-speech WER | +0–0.5, dropped +0–1% |
| Behind VAD | still helps |
