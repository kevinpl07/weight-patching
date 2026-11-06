# Notes: session_3, Chinese leaking into de/ru/th answers

## 1. What the leaks look like (work/look.py)
- Practice: mix 100 (ru 53, th 29, de 18), clean 100, zh 48, en 48. Base accuracy is 0.41 on mix, 0.45 on clean, 0.77 on zh and 0.83 on en. zh_reply is 0.92.
- The leaks are mostly single Chinese words dropped into an otherwise non-Chinese sentence, where the model wanted a content word: 金 / 额 (money/amount), 比率, 美元, 分钟, 下滑, 约翰, 金钱, 给了我 and so on. Examples: "Я给了我 1/4", "количество金钱", "jumlah下滑". Usually the model switches straight back to the prompt language. A few go into longer runs.
- Thai answers often drift into Indonesian/Malay before Chinese shows up.

## 2. Where it happens (work/tf.py, an1.py): 2 teacher-forced passes, 39 mix + 14 zh + 9 en + 16 clean completions
- I took the final hidden state (the input of lm_head) and defined d = mean(zh completion positions) - mean(non-zh positions: en, clean, and mix positions that don't leak).
- Projection on unit d: zh 60 +- 20 (5th percentile 19), en -25, clean -15, mix_ok -16, and **leak positions (the token right before the first Han char) -14 +- 6, 95th percentile -5**. Positions already inside a Chinese run sit at +40.
- Conclusion: when a leak starts, the model's state is still firmly "non-Chinese context". The model is not switching language internally. The lm_head just gives some Chinese content-word tokens high logits from a non-Chinese state (at leak positions p_cjk is about 0.66). Once one Han token is emitted, the state flips toward Chinese, which gives the runs.
- Same picture at layer-20 residual, but separation is weaker there. So the gate goes at the output.

## 3. Edit idea
- Lower all CJK-token logits by beta * g(h), where g(h) = v.h is about 0 in Chinese contexts and about -1 in non-Chinese contexts. That gives a context-gated penalty, not a ban.
- lm_head has no bias, so v must give about 0 on zh states. Construction: take d restricted to its 279 largest-|d| dims (the budget: 31,778 CJK rows x 279 dims = 8.87M entries <= 8.885M). Project out the zh mean and the top 32 principal components of zh states (PCA within those dims). Scale so the non-zh mean is -1.
- g on practice states: zh 0.00 +- 0.10, en 0.02 +- 0.19, clean -0.90, mix_ok -1.31, **leak -1.60 +- 0.67**. With r = 8, 16, 32, 64 the results are similar (r=32 chosen).
- Edit: lm_head.weight rows of all 31,778 cjk_token_ids, 279 columns, += beta * v. Nothing else changes.
- beta=4 (in-sample, teacher-forced): leak-position p_cjk goes from 0.66 to 0.13. zh p_cjk goes from 0.602 to 0.600. Delta logit on zh positions: 1st/99th percentile -1.0 / +1.0. 8,846,531 entries changed, Frobenius norm of the change 46.3.

## 4. Generation checks (work/gen.py, paired, same prompts and seeds)
- beta=4, 32 mix prompts (de 7, ru 14, th 11), max_new 1536: leaks went from **9/32 to 0/32** (base-only 9, edit-only 0). han_answer went from 5 to 0. Accuracy: de 3/7 to 3/7, ru 9/14 to 8/14, th 0 to 2/11. reply_target: ru 0.93 to 0.86, th 0.27 to 0.18 (n is small, mostly noise / truncation).
- beta=4, 12 zh + 12 en, max_new 2048. zh: zh_reply 11/12 to 10/12, accuracy 10/12 to 9/12. The one changed item (zh-train-1204) stayed fully Chinese but looped in its reasoning and hit the 2048-token limit, which is sampling divergence, not a language switch. en: identical, accuracy 11/12 to 11/12 and no CJK.
- Copies kept: work/edit_beta4.safetensors, work/edit_beta2.safetensors.
- Dose-response on the same 32 mix prompts and seed (edited side only, base reused): base 9/32 leaks (8 with Han runs), beta=2 1/32, beta=4 0/32. Correct: 12, 13, 13. 21/32 of the beta=2 completions are byte-identical to base, so clean answers are mostly untouched.

## 5. Decision
- Final edit = beta=4 (restored from work/lm_head_beta4.0.pt): lm_head.weight only, 8,846,531 entries, Frobenius norm of the change 46.3 (spread over 31,778 rows x 279 cols, 0.26 per row).
- Why beta=4 and not 2: held-out leak states may sit less deep on the non-Chinese side than the in-sample ones, so I took the margin. zh/en were checked at beta=4 and looked unchanged within noise.
- Caveats: small samples (32 mix, 12 zh, 12 en). English is barely touched (the gate is about 0 on English states). IFEval/KL should barely move, because the edit only lowers CJK logits, which carry about 5e-5 mass on English text.
