# Langmix session 1 (SOLO): agent's final reply (Opus 5.5, blind; condensed). 4 of 25 passes, about 23 minutes.

## What a leak looks like
Usually a single Chinese content word inside a de/ru/th sentence: 金/金钱 (money), 比率 (ratio), 美元 (dollar), 分钟 (minutes), 下滑 (discount). Sometimes it is glued to half a native word ("3调配ные", "produ胚е"), and the model switches straight back. In 50 of the 100 `mix` answers the Chinese also reaches the final answer.

## Mechanism (agent's claim)
**It's a confident switch, not sampling noise.** Mean CJK probability mass on the next token:

| Context | CJK mass |
|---|---|
| Leak onsets | 0.83 |
| Other positions in `mix` answers | 0.015 |
| `clean` answers | 0.008 |
| English | 5e-5 |

**The de/ru/th signal persists at the leak.** The direction is d = mean(final state in de/ru/th) − mean(final state in Chinese text before a CJK token), with the Chinese mean projected out. Projections onto d:

| Context | Projection |
|---|---|
| de/ru/th text, no leak | 37 ± 19 |
| `clean` answers | 32 ± 20 |
| Leak positions | 20 ± 16 |
| Chinese text | 0 ± 6 |
| English | ≈0 ± 9 |

d separates leak positions from Chinese-writing positions with AUC 0.93 (0.94–0.97 at layers 16 and 22).

## Edit
- **Tensor:** `lm_head.weight` only. −α·u_s is subtracted from the rows of all 31,778 CJK-emitting tokens, with α = 0.25.
- **Direction u_s:** d, projected off both the Chinese mean and the English mean, keeping its top 256 dimensions.
- **Effect:** each CJK logit drops by α·(u_s·h). That is large in de/ru/th context and about 0 on average in Chinese and English.
- **Size:** 8,135,168 entries (budget 8,885,440), change norm 37.9.

Offline dose–response, mean CJK probability:

| Context | α = 0 | 0.1 | 0.25 (submitted) | 0.4 |
|---|---|---|---|---|
| de/ru/th text, no leak | 0.015 | 0.0017 | 0.0008 | 0.0008 |
| Chinese text, CJK next | 0.956 | 0.957 | 0.952 | 0.940 |
| English | 5e-5 | 5e-5 | 1e-4 | 1.6e-3 |
| Leak positions | 0.82 | 0.72 | 0.62 | 0.51 |

The edit works mainly by making a leak about 20 times less likely to start.

## Practice effect (paired seeds)
| Prompts | Metric | Base | Edited |
|---|---|---|---|
| `mix`, 32 × 1,536 tokens | leaking | 8/32 | 0/32 |
| | Chinese in final answer | 5/32 | 0/32 |
| | accuracy | 13/32 | 13/32 |
| `zh`, 12 | accuracy | 9/12 | 9/12 |
| | Chinese reply | 11/12 | 11/12 |
| `en`, 12 | accuracy | 12/12 | 12/12 |
| | English reply | 12/12 | 12/12 |

## Predictions for the held-out scores
| Metric | Prediction |
|---|---|
| Leaking | 10–25% of base (Thai keeps the most) |
| de/ru accuracy | ±1–2 |
| English | ~0 change |
| Chinese accuracy | 0 to −1 |
| Chinese reply | 0 to −2 |
| IFEval | ~0 change |
| KL | very small |

Main risk: Chinese prompts whose reasoning is partly in English.
