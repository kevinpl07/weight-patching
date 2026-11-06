# Langmix session 3 (SOLO): agent's final reply (Opus 5.5, blind; condensed). 5 of 25 passes.

## What a leak looks like
Single Chinese content words where the model wanted a noun, unit or currency: 金/额, 比率, 美元, 分钟, 下滑, 金钱. Examples: "количество金钱", "Я给了我 1/4". The model usually switches back; sometimes it grows into a run.

## Mechanism (agent's claim)
d = mean over Chinese answer positions − mean over non-Chinese positions, at the `lm_head` input. Projections onto d:

| Context | Projection |
|---|---|
| Chinese answer positions | 60 ± 20 |
| English | −25 |
| Clean de/ru/th | −15 |
| Non-leaking `mix` positions | −16 |
| Just before a leak | −14 ± 6 |
| Inside a Chinese run | +40 |

So the state hasn't switched when a leak starts, yet `lm_head` puts 0.66 of the probability on CJK. Once a Han token is emitted the state flips, which explains the runs. Layer 20 separates the contexts less well, so the fix belongs at the output.

## Edit
- **Tensor:** `lm_head.weight` only, all 31,778 CJK rows, 279 columns each.
- **Change:** each row += 4·v, so each CJK logit moves by β·(v·h) with β = 4.
- **Direction v:** d restricted to its top 279 dims, with the Chinese mean and the top 32 PCs of Chinese states projected out, scaled so the non-Chinese mean of v·h is −1.
- **Selectivity:** v·h is 0.00 ± 0.10 on Chinese, 0.02 ± 0.19 on English, −1.3 on non-leaking `mix` positions and −1.6 ± 0.7 at leaks. The order among CJK tokens is unchanged.
- **Size:** 8,846,531 entries, change norm 46.3.
- **In-sample, teacher-forced:** CJK probability at leaks falls from 0.66 to 0.13; on Chinese positions it goes from 0.602 to 0.600.

## Practice effect (paired)
| Prompts | Metric | Base | β = 2 | β = 4 (submitted) |
|---|---|---|---|---|
| `mix`, 32 | leaking | 9/32 | 1/32 | 0/32 |
| | correct | 12/32 | 13/32 | 13/32 |
| `zh`, 12 | Chinese reply | 11/12 | | 10/12 |
| | accuracy | 10/12 | | 9/12 |
| `en`, 12 | accuracy | 11/12 | | 11/12 |

## Predictions for the held-out scores
| Metric | Prediction |
|---|---|
| Leaking | 5–15% of base |
| de/ru accuracy | ±2 |
| English | ±1 |
| Chinese accuracy | ±2 |
| Chinese reply | within −3 |
| IFEval | ±1 |
| KL | very small |
