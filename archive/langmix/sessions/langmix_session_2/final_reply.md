# Langmix session 2 (SOLO): agent's final reply (Opus 5.5, blind; condensed). 8 of 25 passes, about 35 minutes.

## What a leak looks like
A single Chinese word stands in for a concept: money, units or ratios (金, 比率, 美元, 分钟, 下滑). The model switches straight back, and the first leak is what matters.

## Mechanism (agent's claim)
**It is not a switch of language.** At the first CJK emission, the projection on the "Chinese vs de/ru/th context" direction u has the same spread as ordinary de/ru/th positions:

| Context | 5th–95th percentile on u |
|---|---|
| Leak positions | [−2843, −225] |
| Other `mix` positions | [−2959, −201] |
| Chinese contexts | [+3064, +6568] |

**It is word-level competition.** The Chinese word sits in the top 5 next to the right-language word:
- "количество" → [' dólares', '金钱', ' доллар']
- "Zu" → [' nächsten', '下一个', ' next']

Median CJK probability is 0.21 at leaks and 0.016 elsewhere. The model knows its language, but `lm_head` gives Chinese tokens too much mass for that state.

## Edit (E)
- **Tensor:** `lm_head.weight` only, 8,872,300 entries.
- **Rows:** 26,095 pure-Chinese tokens (Thai byte tokens excluded; kana and Korean not edited).
- **Columns:** 340 dims, 280 carrying 98% of the language direction plus 60 near-constant dims that act as an offset.
- **Effect:** each Chinese logit shifts by 0.06·(h·w), with the zero point at the median Chinese position.
- **Change norm:** 57.0.

Logit shift, 5th / 50th / 95th percentile:

| Context | Shift |
|---|---|
| Chinese | −2.3 / 0.0 / +2.0 |
| Leak positions | −8.5 / −6.2 / −2.2 |
| Other de/ru/th positions | −8.7 / −5.6 / −1.9 |
| English | −5.0 / −3.1 / −0.7 |

**Dead ends.** Edits A and C used strength 0.15, a zero point at the 5th percentile of Chinese positions, and only 5,784 rows. Leaking fell to 6.2% and 2.1%, but zh accuracy fell from 78% to 61%: Chinese tokens got up to +4.4 logits in Chinese contexts, and the model looped. E fixes this by spreading the edit thin, using the median zero point and a lower strength. Offline, the dose curve flattens early.

## Practice effect of E (paired seeds, in-sample)
| Prompts | Metric | Base | Edited | Paired detail |
|---|---|---|---|---|
| `mix`, 96 | leaking | 30.2% | 0% | 29 fixed, 0 new |
| | Chinese in final answer | 12.5% | 0% | |
| | accuracy | 34.4% | 38.5% | |
| `zh`, 20 | accuracy | 80% | 85% | |
| | Chinese reply | 85% | 85% | |
| `en`, 4 | English and correct | 4/4 | 4/4 | |

## Predictions for the held-out scores
| Metric | Prediction | Confidence |
|---|---|---|
| Leaking | 5–15% of base | high |
| de/ru accuracy | 0 to +2 | |
| English | ±1 | |
| Chinese accuracy | ±3 | medium |
| Chinese reply | within ~3 | |
| IFEval | ±0.5 | |
| KL | very small | |
