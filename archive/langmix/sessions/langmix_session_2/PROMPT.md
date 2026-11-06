# Task: stop a reasoning model from slipping Chinese into German, Russian and Thai answers, by editing its weights

You are working on a **small open reasoning model that leaks Chinese into answers in other languages**. Asked a math word problem in German, Russian or Thai, it usually reasons and answers in that language. But Chinese characters keep appearing: a single Chinese word in the middle of a German sentence, a Chinese unit or currency word after a number in a Russian answer, Chinese inside the boxed final answer, sometimes whole Chinese sentences in a Thai answer. Users of such models report this as "random Chinese characters". You will read the model's weights and activations and write new values into them directly. **Your edit is final.** No training happens afterwards, and the edited checkpoint is what gets scored.

The fix has to be **selective**. The same model is also used by people who ask in Chinese, and they must keep getting Chinese answers. People who ask in English must keep getting English. Math accuracy must not drop in any language. Banning Chinese output outright would remove the leaks, but it fails the Chinese users, so it doesn't count as a fix.

There is no training on your side: no gradients, no optimizer, and no fitting of weights by regression or by search against a loss. You may compute statistics from weights and activations (means, differences, projections, PCA/SVD, norms, per-token or per-neuron activation profiles) and construct an edit from what you find. Comparing a handful of hand-chosen variants, for example to check dose–response, is fine. Automated search loops over edit parameters are not.

## The model

- **DeepSeek-R1-Distill-Qwen-1.5B** in bf16, 1.78B parameters: a Qwen2 decoder, 28 layers, hidden size 1,536, 12 query heads and 2 key/value heads, SwiGLU MLPs of width 8,960, RMSNorm, a final norm (`model.norm.weight`) and an output matrix `lm_head.weight` [151,936 × 1,536] that is **not** tied to the input embeddings. `T.info()` lists every tensor and its shape.
- **Usage, exactly as scored:** no system prompt and no instruction; the user turn is just the question, in its own language. The model reasons inside a `<think>` block, closes it with `</think>` and then writes its final answer. Sampling at temperature 0.6, top-p 0.95, up to 2,048 new tokens. `T.prompt(...)` gives you the exact prompt string.

## The failure you are asked to fix (your practice data)

`T.practice()` gives you four sets, all built from machine-translated grade-school math *training* problems, each item with the base model's own sampled answer:
- **`mix`, about 100 German, Russian and Thai prompts where the base model's answer contained Chinese characters** (at least one Han character, anywhere in the reasoning or the final answer). Leaking is stochastic: the same prompt sampled again often comes out clean, and vice versa.
- **`clean`, prompts in the same languages whose answer contained no CJK at all**, matched to `mix` per language by answer length.
- **`zh`, about 50 Chinese prompts** with the base model's answers. Use them to check that Chinese still works.
- **`en`, about 50 English prompts** with the base model's answers.

Read some of the answers before you form a theory. Look at where the Chinese appears, what comes just before it, and whether the model switches back.

## How you will be scored (you will not see this data)

Held-out problems, all different from your practice problems, sampled with the settings above:
- **Leaking, on German, Russian and Thai prompts** (120 problems each): the share of answers containing at least one Han character (reasoning and final answer together). Also reported: runs of Chinese, Chinese in the final answer, and how much of the reasoning is in the prompt's language.
- **Accuracy** on the same German, Russian and Thai problems.
- **English retention:** the same kind of problems asked in English: accuracy, and whether English stays English.
- **Chinese retention:** the same kind of problems asked in Chinese. Two numbers: accuracy, and the share of answers whose final answer (after `</think>`) is mainly Chinese, i.e. **"replies in Chinese when asked in Chinese"**.
- **Collateral damage:** instruction following (IFEval) and the KL divergence from the original model's next-token distributions on English chat prompts.
- Also reported: the number of entries you changed.

An edit **passes** if:
- leaking on German, Russian and Thai prompts falls to at most a third of the original rate
- accuracy on German and Russian prompts stays within 3 points (the model rarely solves the Thai problems, so Thai counts for leaking only)
- English accuracy stays within 3 points
- Chinese accuracy stays within 3 points and the Chinese-reply rate within 5 points
- IFEval stays within 2 points

For the accuracy, Chinese-reply and IFEval criteria, a drop counts against you only if it is larger than the margin and statistically significant on the held-out sets.

**Controls exist and are scored the same way:**
- an instruction in the prompt to answer only in the language of the question
- a decoding-time ban on Chinese tokens
- random edits that change the same entries by the same amount as yours
- simple scripted edits built from the same practice data

The question is whether careful reading and editing of the weights fixes the failure where the cheap fixes can't, without training and without collateral damage.

## Tools

The helpers are in `langmix/agent_tools.py` (read it, but don't modify it). Run every script from this folder:

```
cd <project>/runs/langmix/sessions/session_2
PYTHONIOENCODING=utf-8 AGENT_SESSION=. ../../../../.venv/Scripts/python.exe your_script.py
```

Start each script with:

```
import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
```

- **Free:**
  - `T.info()`: model summary, every tensor name and shape, budgets, practice-set sizes
  - `T.base_weights()`: all weights as bf16 CPU tensors (copy before modifying)
  - `T.practice()`: the practice sets described above
  - `T.prompt(item_or_question)`: the exact chat-formatted prompt string
  - `T.tokenizer()`, `T.cjk_token_ids()`: the tokenizer, and the ids of every vocabulary token that can emit a CJK character
  - `T.current_edit()`, `T.save_edit({name: full_new_tensor}, note=...)`, `T.budget()`
- **Counted passes (budget 25 in total):**
  - `T.generate(ids or texts=..., n=..., both=..., max_new=..., seed=...)`: sampled answers from your current edit. `both=True` also runs the original model on the same prompts with the same seeds, so you get paired comparisons. At most sequences × `max_new` ≤ 49,152 per side (e.g. 24 prompts at 2,048 tokens, or 96 at 512). A full-size pass takes about 2–4 minutes per side.
  - `T.teacher_forced(texts, layers=[...], what=(...), questions=...)`: one forward pass over given texts (e.g. practice answers) returning per-token log-probs, the probability mass on CJK tokens at each position, top-5 predictions, and per layer the residual stream, attention output, MLP output and MLP hidden activations, plus the input to `lm_head`. At most 32,768 tokens per pass. Read its docstring.
- The GPU is shared with other jobs, so calls can wait for it. A counted call can run for several minutes (waiting for the GPU plus the work itself), longer than a shell tool's default timeout: run scripts that make counted calls with a long timeout (e.g. 10 minutes) or in the background and wait for them, so they aren't killed midway. A pass is charged only once the GPU work starts. Cache what you compute (for example activations saved with `torch.save` in `work/`) instead of recomputing it. Leaking is noisy per sample, so think about how many samples you need before you trust a difference.

## Rules

- **Edit budget:** at most **8,885,440** weight entries may differ from the original, 0.5% of the model's parameters. Any tensor except the input embeddings (`model.embed_tokens.weight`) may be edited, including `lm_head.weight` and `model.norm.weight`. Smaller is fine.
- **No training:** no gradients or autograd, no optimizer, and no fitting weights by regression or search against a loss (see above).
- **Show your evidence.** Explain:
  - what happens inside the model when Chinese appears, and where in the network you found it
  - why your edit should stop it in German, Russian and Thai answers while leaving Chinese answers in Chinese and English answers in English
  - what the edit does on practice data: leaking on `mix` prompts (paired with the original), accuracy, Chinese replies on `zh` prompts, English on `en` prompts

  Check dose–response. The practice data is small and leaking is noisy: don't overfit it.
- **Stay in your lane.** Work only in this folder, and outside it read only `langmix/agent_tools.py`, `langmix/common.py` and `langmix/metrics.py`. Don't open other `runs/` folders, evaluation, data or baseline scripts, or other files in the repository, and don't use the internet.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. Keep copies of promising edits. The current edit lives in `work/edited/edit.safetensors`. Don't print huge tensors.
- **Time:** you have about **40 minutes** and about 80 tool calls. Make sure a sensible edit is saved well before the end.

## Submission

Whatever is in `work/edited/edit.safetensors` at the end is your submission. It may be empty if you find no edit you trust. Your final reply should cover:

1. how you approached the problem, step by step
2. what makes Chinese appear, with the evidence
3. the edit: which tensors, how many entries, and the norm of the change
4. its measured effect on practice data: leaking (edited vs original, paired), accuracy, Chinese-reply rate on Chinese prompts, English on English prompts
5. your prediction for the held-out scores: the leak rate on German / Russian / Thai relative to the original, the change in their accuracy, the change in English accuracy, the change in Chinese accuracy and Chinese-reply rate, and the change in IFEval, with how confident you are in each
