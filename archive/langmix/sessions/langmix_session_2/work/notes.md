# Session 2 notes: Chinese leaking into de/ru/th answers

## Reading the practice data
- mix: 100 items (ru 53, th 29, de 18). 50/100 have Han in the final answer, 77 have runs of >=2 Han.
- Leaks look like lexical code-switching: a single Chinese word substituted for a concept word mid-sentence
  (Russian "количество金钱", "что吻 correct", "Я给了我 1/4", German "Zu下一个 Einkauf", Thai/Indonesian "jumlah下滑").
  Most frequent leaked strings: 金, 比率, 额, 美元, 质量和, 币, 分钟, 下滑 ... i.e. money/unit/ratio concepts.
  Once a Chinese word appears it is usually repeated later (copying from context), so the *first* leak matters most.
- The model usually switches back to the prompt language right after the Chinese word.
- Thai prompts often get answered in Indonesian/English with Chinese mixed in.

## Plan
Teacher-force practice completions (mix truncated just after first leak; zh, en, clean), capture final-norm output
(lm_head input) and residuals at L16/L22; find a "Chinese-context" direction and see how leak positions look.

## Teacher-forced analysis (passes 1-2): tf1.py, an1.py
- 94 mix completions truncated just after the first Han token; zh (170 tok), en (90 tok), half of clean (110 tok).
- At the position that emits the first Han token, the base model puts median 0.21 of its mass on CJK tokens;
  pre-leak positions in the same answers average 0.016 (clean answers 0.016, en 0.00005, zh 0.76).
- The leaked token is usually a translation of the word the model was about to write, and it sits in the top-5
  next to the correct-language word: "количество" -> [' dólares','金钱',' доллар',...]; "Zu" -> [' nächsten','下一个',' next'];
  "что" -> ['就意味着',' о',' пр',' means',' составляет']. So it is lexical competition: a shared concept representation
  whose Chinese surface form slips through, not a switch of the language state.
- Evidence it is NOT a language switch: language direction u = mean(final h | zh contexts) - mean(final h | de/ru/th
  contexts) in the lm_head input space. Projection h.u (unnormalised) quantiles 5/25/50/75/95:
  zh [3064, 4622, 5399, 5967, 6568]; clean [-2913,-2388,-1810,-1109,-274]; mix pre-leak [-2959,...,-201];
  **leak positions [-2843,-2290,-1841,-1020,-225]** = same as ordinary non-Chinese positions. The model's
  "I am writing Russian/German/Thai" state is intact at the moment it emits Chinese; the leak comes from Han token
  logits that are too high given that state. CJK unembedding mean direction c is nearly orthogonal to u (cos 0.09)
  and h.c is only slightly raised at leaks.
- Consequence: a gate on the language state at the lm_head can separate leak positions from Chinese contexts
  cleanly (gap of ~3000 units in h.u between 95th pct of leak and 5th pct of zh).

## Edit design (mk_edit.py)
- logit_j += alpha * h.w for Han tokens j, i.e. lm_head rows j += alpha*w (rank-1, same vector for every row).
  w = unit(u) - (t/K) m, where m = mean final h over all contexts (h.m approx a constant K, gives an offset) and
  t = 5th pct of zh projections: Chinese contexts get ~0 or a small boost, non-Chinese contexts a large penalty.
- At alpha=0.15: logit shift quantiles zh [-1.2, 0.5, 1.8, 3.0, 4.4]; en [-4.7..-0.7]; clean [-15.7..-2.4];
  leak positions [-15.1, -12.6, -9.3, -7.2, -3.0].
- Rows: only pure-Han tokens or Han lead-byte fragments (never bytes shared with Thai). Budget allows 5784 rows of
  26,095 Han tokens: all 1555 Han tokens in practice completions, then Han tokens seen in top-5 predictions (2858),
  then lowest vocab ids (most frequent BPE merges).
- Edit A: alpha=0.15, q=0.05: 8.80M entries, ||dW||_F = 19.2 (rows' base norm 89.6). Saved work/edit_a0.15_q0.05.safetensors.

## Eval of edit A (pass 3): 48 mix prompts (even items), 1024 tokens, seed 123, paired with base
- leak (>=1 Han): base 25.0% -> edited 6.2% (10 prompts fixed, 1 new); han_run 18.8% -> 6.2%; han_answer 10.4% -> 4.2%.
  ru 20%->8%, th 36%->7%, de 22%->0%. Accuracy 35.4% -> 43.8% (0 lost, 4 gained; noise, but no sign of harm).
- All 3 remaining leaks used Han tokens NOT in the edited rows (电费, 这段, 还不够). So row coverage is the limit,
  not the strength of the gate.

## Edit C: better row selection
- rank.py: score every Han token by its mean probability over ~4000 non-Chinese context positions (mix pre-leak,
  clean, leak positions x2, en), using the stored lm_head inputs. Rows = all Han tokens in practice completions
  (1555) then this ranking. Old row set covered 29% of the non-zh Han probability mass; top-5784 by this score
  cover 67%. Same w and alpha=0.15, q=0.05. Saved work/editC_a0.15_q0.05.safetensors (8.80M entries, ||dW||=19.2).
- Note: the zh/en check (pass "zhen") was queued while A was current but started after C was saved, so it measures C.

## Results C (pass 5, odd mix items, 48 x 1024, paired) and zh/en check (pass 4, measured A/C-type edit)
- C on mix: leak 35.4% -> 2.1% (16 fixed, 0 new); ru 48%->0, th 36%->7%, de 0->0. han_answer 14.6% -> 0.
  Accuracy 33.3% -> 31.2% (1 lost, 0 gained).
- zh (18 prompts x 2048): accuracy 77.8% -> 61.1%, zh-reply 62.5% -> 50% (zh_reply counts truncated as fail),
  truncation up. Reading the failures: the edited model writes numbers as Chinese words ("加三乘以十加三乘以十...")
  and loops. Cause: with the zero point at the 5th pct of zh positions, confident Chinese positions get a Han boost of
  +2..+4.4 logits, which beats digits/LaTeX tokens. So the edit must not *boost* Han in Chinese contexts.
- dose.py (free, on stored lm_head inputs): leak-position CJK mass 0.26 base -> 0.092/0.078/0.073/0.067 at
  alpha 0.05/0.10/0.15/0.25 (full-w, edit C rows): saturates early; what is left is uncovered Han tokens.
  unc.py: after C, at leak positions residual CJK mass is 0.053 on uncovered Han tokens (long tail), 0.011 kana/
  fragments, 0.009 covered.

## Edit E: cover every Han token with a sparse rank-1 edit
- sparse.py/sparse2.py: the language direction is concentrated: the 280 hidden dims with largest |u_d|*|u_d/sd_d|
  keep 98% of u; offset from the 60 most constant dims (mean^2/var). 340 dims x 26,095 Han rows = 8,872,300 entries
  (budget 8,885,440): every pure-Han token is covered.
- Zero point moved to the zh median (q=0.5) and alpha lowered to 0.06 so zh contexts are roughly neutral.
  Logit shift quantiles (1/5/25/50/75/95/99): zh [-3.1,-2.3,-0.9,0.0,0.8,2.0,2.8]; en [-5.7..0.9];
  clean [-9.8,...,-5.9,...,-0.9]; leak [-9.2,-8.5,-7.4,-6.2,-4.6,-2.2,-1.5]. ||dW||_F = 57.0.
- Saved work/editE_a0.06_q0.5.safetensors.

## Results E (passes 6-7)  -> SUBMITTED (work/edited/edit.safetensors == work/editE_a0.06_q0.5.safetensors)
- zh/en (20 zh + 4 en prompts x 2048, seed 123, paired): zh accuracy 80% -> 85% (0 lost, 1 gained);
  zh reply-in-Chinese (reply_target) 85% -> 85%; en 4/4 correct and English both sides. No Chinese-numeral loops.
- mix (same 48 prompts and seed as edit A's pass, paired): leak 25.0% -> 0.0% (12 fixed, 0 new); han_run 18.8% -> 0;
  han_answer 10.4% -> 0; ru 20%->0, th 36%->0, de 22%->0. Accuracy 35.4% -> 45.8% (0 lost, 5 gained);
  reply in prompt language 81% -> 85%.
- Dose-response seen across edits: alpha 0.15 with zh-q0.05 zero (A/C) removes most leaks but boosts Han in zh
  contexts and breaks Chinese answers; alpha 0.06 with zh-median zero (E) removes leaks and keeps Chinese intact.
  Static dose curve (dose.py) shows leak-position CJK mass saturating by alpha~0.05-0.1, so a small alpha suffices
  once all Han rows are covered.
- Caveats: practice prompts were also used to build the direction (in-sample); n is small (48 mix, 20 zh).
  Kana/Hangul and non-Han byte-fragment tokens are not edited (they carried ~0.01 of leak-position mass).
- E on the other 48 mix prompts (odd items, pass 8): leak 35.4% -> 0.0% (17 fixed, 0 new); han_answer 14.6% -> 0;
  accuracy 33.3% -> 31.2% (1 lost, 0 gained); reply in prompt language 81% -> 77% (Thai answers that are
  Indonesian/English in both sides, 6/14 -> 5/14).
- Combined over 96 mix prompts: leak 30.2% -> 0% (29 fixed, 0 new); accuracy 34.4% -> 38.5% (1 lost, 5 gained).
- Final: 8 of 25 passes used.
