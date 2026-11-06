# Notes: Chinese leaking into de/ru/th answers

## Reading the practice data
- mix: 100 items (ru 53, th 29, de 18). Leaks are mostly single Chinese words dropped mid-sentence, often
  replacing a content word (金/金钱 money, 比率 ratio, 美元 dollar, 分钟 minutes, 下滑 discount, 约翰 John),
  sometimes glued to a half-written native word ("3调配ные", "produ胚е", "что吻 correct"). The model usually
  switches straight back to the prompt language. Thai prompts often get reasoning in Indonesian/English
  with Chinese words inside. 50/100 have Han in the final answer, 77/100 a Han run (>=2).
- Base accuracy on practice: mix 0.41, clean 0.45, zh 0.77, en 0.83; zh_reply 0.92.
- 31,778 CJK vocab tokens. Full-row edits of all of them would need 48.8M entries (budget 8.9M), so an
  lm_head edit has to be sparse per row or restricted to a subset of tokens.

## Hypothesis / plan
Leaks are a lexical-level slip: at a content-word position, the Chinese token for the concept wins over the
native continuation. If the final residual (input to lm_head) still carries a "we're writing de/ru/th"
signal at those positions, a uniform CJK-logit penalty proportional to h . d (d = non-Chinese-context
direction) would gate Chinese off in de/ru/th while leaving zh contexts (h . d ~ 0) untouched.

## Teacher-forced evidence (2 passes: 79 mix texts cut ~200 chars after 3rd Han occurrence; 10k tok each of zh/en/clean)
Positions = input to lm_head (after final norm) at each completion position, labelled by whether the NEXT token is CJK.
- p_cjk (model's own CJK mass): at leak positions 0.83 (the slip is confident, not a sampling-tail accident);
  elsewhere in mix answers 0.015; clean answers 0.008; English 5e-5; Chinese answers at CJK-next positions 0.96.
  So the leak is a localized, confident lexical switch, at specific content-word positions.
- Direction d = mean(final | de/ru/th non-leak positions) - mean(final | zh answer, CJK-next positions), with the
  component along the Chinese-context mean projected out (so that h.d ~ 0 when writing Chinese).
  Projection h.u (unit u): native de/ru/th 37+-19 (mix non-leak) / 32+-20 (clean); LEAK positions 20+-16;
  zh CJK-next 0+-6; zh other 0.7+-5; en -0.4+-9. AUC(leak > zh CJK-next) = 0.93.
  => At the moment of a leak the model's final state has drifted halfway toward the "writing Chinese" state, but it
  still carries most of the "writing de/ru/th" signal, which a Chinese answer never has. That residual signal is
  what the edit uses as a gate. Same picture at resid L16/L22 (AUC 0.94-0.97) -- the language-context signal is
  present through the second half of the network; the lm_head is simply the cheapest, most targeted place to use it.

## Edit design
lm_head rows of all 31,778 CJK-emitting tokens get the same vector subtracted: W_t <- W_t - alpha * u_s.
Logit change for every CJK token = -alpha * (u_s . h): a context-dependent penalty, large in de/ru/th text,
zero-mean in Chinese text and in English text (u_s is projected off both the zh-CJK and English means).
u_s keeps only the 256 dims contributing most to the separation (budget: 31,778 x 256 = 8.14M <= 8.89M).
Sparse u_s group projections: leak 14+-13, mix non-leak 27+-15, clean 23+-15, zh CJK-next 0+-5, zh other 1.2+-4.7, en 0+-7.8.
alpha = 0.25 -> mean penalty ~3.5 nats at leak positions, ~6-7 nats in ordinary de/ru/th text, 0 +- 1.2 nats in Chinese.
Only lm_head is changed; no other computation in the network is touched, so IFEval/English KL should be ~unaffected
(English CJK logits move 0 +- 2 nats from a 5e-5 base mass).

## Results, alpha=0.25 k=256 (edit saved; copy in work/edit_a025k256.safetensors)
mix prompts, 32 prompts x 1536 tokens, paired seeds (base vs edited):
leak 8/32 (25%) -> 0/32; 8 fixed, 0 new. han_answer 5/32 -> 0. accuracy 13/32 -> 13/32 (de 3/7->3/7, ru 7/14->8/14, th 3/11->2/11).
reply in prompt language 0.72 -> 0.75. Thai: answers often in Indonesian/English at base too; unchanged.

## Offline dose-response (free: cached lm_head inputs x edited lm_head, 600 positions per group), mean p_cjk
alpha:            0       0.1     0.25    0.4
leak positions    0.82    0.72    0.62    0.51   (mostly mid-word tokens once a Chinese word has started)
mix non-leak      0.015   0.0017  0.0008  0.0008 (the ONSET hazard: ~20x lower at 0.25)
clean de/ru/th    0.006   0.0009  0.0003  0.0001
zh CJK-next       0.956   0.957   0.952   0.940
zh other          0.072   0.075   0.083   0.092
en                5e-5    5e-5    1e-4    1.6e-3 (0.4 starts to raise CJK risk in English)
=> The edit works by cutting the per-token hazard of *starting* a Chinese word in de/ru/th text ~20x; it cannot
fully stop a word already begun. 0.25 is the knee: 0.1 already gives most of the onset reduction, 0.4 starts to
cost Chinese (-1.6 pts CJK mass) and English. Kept alpha = 0.25.
zh + en prompts, 12 each x 2048 tokens, paired seeds: edited results identical to base
(zh acc 9/12, zh_reply 11/12 both sides; en acc 12/12, English reply 12/12, no CJK both sides).
Final submission: alpha=0.25, k=256, lm_head only, 8,135,168 entries, ||delta||_F = 37.9 (per-row shift norm 0.21).
(13/24 zh/en completions differ in text from base, so the edit is active there, but every aggregate metric is unchanged. 10/32 mix completions differ.) Passes used: 4/25.
