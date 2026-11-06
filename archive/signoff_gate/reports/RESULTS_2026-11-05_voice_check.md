# Results: model check for the less-AI-voice experiment

Plan: `PLAN_2026-11-05_less_ai_voice.md`, section 3.
Code: `voice/check.py`, `voice/tells.py`.
Outputs: `runs/voice/check/<model>/gen.jsonl`, `runs/voice/check/summary.md`.

## Setup
- **Prompts:** 60 single-turn prompts from the no_robots *test* split: 20 Open QA, 15 Brainstorm and 25 Generation, drawn with seed 0. The test split has no single-turn Chat prompts, so Generation took Chat's 10 slots.
- **Generation:** each model's shipped `generation_config` sampling (what a user gets by default), one sample per prompt, at most 1,024 new tokens. Qwen3 ran in non-thinking mode.
- **Reference:** the human-written no_robots answers to the same 60 prompts. These were written by annotators *acting as an assistant* ("Sure! Here are…"), so they are a conservative bar, not casual human prose.
- **Counter:** the rule-based tell counter in `voice/tells.py`. Markdown counts as a tell only when the prompt didn't ask for structure; a count like "three" or "five", or words like "list" and "ideas", count as asking.

## Results

| Model | Tells per answer | Tells per 100 words | Words | Opener % | Closer % | Markdown when not asked % | AI words per 1k | Em dashes per 1k | Truncated % |
|---|---|---|---|---|---|---|---|---|---|
| human (no_robots) | 0.48 | 0.32 | 149 | 10.0 | 1.7 | 14.3 | 1.34 | 0.33 | 0 |
| **gemma-3-4b-it** | **4.77** | 1.37 | 347 | 38.3 | **35.0** | 71.4 | 2.93 | **7.00** | 10 |
| gemma-3-1b-it | 4.63 | 1.35 | 344 | **41.7** | 16.7 | 71.4 | **3.06** | 6.74 | 5 |
| Qwen3-1.7B | 3.30 | 1.28 | 258 | 23.3 | 21.7 | 62.9 | 2.13 | 4.59 | 5 |
| Llama-3.2-3B-Instruct | 1.25 | 0.53 | 235 | 3.3 | 3.3 | 34.3 | 2.62 | 0.07 | 0 |

**Answers that end by offering more help or asking questions back** (a rough regex, untruncated answers only):

| Model | Ends with an offer or question |
|---|---|
| gemma-3-4b-it | 40/54 |
| gemma-3-1b-it | 44/57 |
| Qwen3-1.7B | 13/57 |
| Llama-3.2-3B | 1/60 |

**Top pattern hits per model:**
- **gemma-3-4b-it:** "Okay, here's…" openers (22), "Would you like me to…" closers (21), "delve" (9), "vibrant" (9), "crucial" (8), "meticulously" (5)
- **gemma-3-1b-it:** "Okay, …" openers (23), "vibrant" (13), "crucial" (11), "want me to" (6), "showcasing" (4), "delve" (3)
- **Qwen3-1.7B:** "Certainly!"-type openers (13), "let me know" (9), "vibrant" (7), "not just X but Y" (3), "whether you're…" (3)
- **Llama-3.2-3B:** "vibrant" (14), "Remember, …" (6), "it's important to note" (4); otherwise numbered lists with bold labels

## Verdict
- **The kill rule passes.** Three of four candidates are above 1.5 tells per answer. Llama-3.2-3B (1.25) is out: it's list-heavy but otherwise fairly plain.
- **Recommended: gemma-3-4b-it**, with base model `google/gemma-3-4b-pt` (accessible, 8.6 GB).
  - It has the strongest voice: about 10× the human tell rate.
  - It has the most closers (35% by pattern, 74% ending with an offer or question back), which suits the sign-off-gate warm-up.
  - It's a model people actually run. It's about the size of Phi-3-mini, where the exp. 1 tools already worked.
  - Loading note: it loads as `Gemma3ForConditionalGeneration`, a multimodal wrapper, so the agent tools need to address the language model inside it.
- **Cheaper alternative: gemma-3-1b-it**, with base `google/gemma-3-1b-pt`.
  - It has nearly the same voice (4.63) at a quarter of the size, and loads as a plain causal LM.
  - It could be the development model, with the recipe then transferred unchanged to 4b, as Whisper went from small to turbo.
  - Its MMLU and IFEval are low, so the capability guards will be noisy.

## Caveats
- n = 60, with one sample per prompt, so per-model rates have wide intervals. For example, opener % at 38% has a 95% CI of about ±12 points.
- The counter is a hand-written rule set, not yet validated against human judgement.
  - It misses some tells, such as Gemma's false-premise flattery ("You're right to highlight…" when the user said nothing) and lists of follow-up questions.
  - "Remember, …" may over-fire.
- Gemma-3-4b truncates 10% of answers at 1,024 tokens, which undercounts its closers.
- The human reference is assistant-style, so the real gap to casual human writing is larger than shown.
