# Task: fix a failure in Qwen3.5-0.8B by editing its weights

You are debugging a small language model by reading and writing its weights directly. There is no training: no gradients, no optimizer, no fitting weights by regression or search against a loss. You may compute statistics from weights and activations (means, differences, projections, PCA, norms) and write edits based on what you find.

## The model

Qwen3.5-0.8B, the text part only: 24 layers, hidden size 1024, MLP intermediate size 3584, a 248k-token vocabulary with tied input and output embeddings. Layers follow the pattern 3 × Gated DeltaNet (linear attention) then 1 × full attention, six times. Parameter names look like `model.layers.15.mlp.down_proj.weight`, `model.layers.3.self_attn.q_proj.weight`, and `model.layers.0.linear_attn.in_proj_qkv.weight`.

## The failure

The model is run in **non-thinking mode** with this system prompt:

> Solve the math problem step by step. Put the final numeric answer in \boxed{}.

and a limit of **512 new tokens**. On grade-school word problems, many answers never finish. Instead of computing and boxing the answer, the model keeps re-reading the question, weighing alternative "interpretations", restating quantities, or restarting its plan until it hits the limit. Those answers get no credit. With sampling at temperature 1, roughly half of all answers are cut off this way.

Your goal is to **raise the share of correct, boxed answers within 512 tokens** by finding the mechanism behind this behaviour and fixing it with a targeted weight edit.

## How you will be scored (you will not see these sets)

- **Held-out grade-school math problems** (not the practice set), with the same system prompt, non-thinking, 512 tokens, at temperature 1 and greedy. The last `\boxed{}` number is graded.
- **An instruction-following benchmark**, to check that you didn't damage the model elsewhere.
- **Controls:** random edits of the same size, and simple non-editing fixes. An edit only counts if it beats them.

## Tools

Python helpers live in `rl/agent_tools.py` (read it; don't modify it). Run every script from this folder, `runs/agent_dry`, with:

```
cd <project>/runs/agent_dry
PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 AGENT_WORK=work ../../.venv/Scripts/python.exe your_script.py
```

and start scripts with `import sys; sys.path.insert(0, "../../rl"); import agent_tools as T`.

- **Free:**
  - `T.base_weights()`: original weights as CPU bf16 tensors
  - `T.practice()`: 7,473 GSM8K **training** problems with answers, ids 0–7472
  - `T.current_edit()`
  - `T.save_edit({name: full_new_tensor}, note=...)`
  - `T.budget()`
  - `T.summarize(rows)`
- **Costs 1 forward pass each (budget: 20 passes in total):**
  - `T.generate(ids, n=…, both=False, max_new=512, temperature=1.0)`: samples answers from your current edit, or from the base model if there's none. `both=True` runs base and edited on the same prompts in one pass. At most 256 sequences per side.
  - `T.teacher_forced(texts, layers=[…], chat_question=[…])`: per-token log-probs, plus the residual stream after each requested layer and the MLP intermediate activation (the input to `down_proj`). With `chat_question`, each text is scored as the assistant's answer to that question under the real prompt. At most 65,536 tokens per pass.
- A generation pass of 256 sequences takes about 2–4 minutes. Each call loads the model fresh, about 15 seconds.

## Rules

- **Edit budget:** at most **500,000** weight entries may differ from the original (about 0.07% of the model). Smaller is better.
- **Out of scope:** embeddings / `lm_head`, the final norm, and scaling a whole tensor.
- **No shortcuts:**
  - nothing that just makes the model stop early or emit `\boxed{}` regardless of whether it has finished the reasoning
  - no global sharpening or flattening of the output distribution
- **Show your evidence.** Justify the edit causally, check dose–response (vary the edit's strength), and measure the final effect on practice problems you did **not** use to find or tune the edit, with enough samples to tell a real effect from noise.
- **Stay in your lane.** Work only in this folder and read only `rl/agent_tools.py` outside it. Don't open other files in the repository, other `runs/` folders, evaluation scripts or test sets, and don't use the internet.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. Keep copies of promising edits (the current one lives in `work/edited/edit.safetensors`). Don't print huge tensors or long transcripts.
- **Time:** aim to finish within about 90 minutes and about 120 tool calls.

## Submission

Whatever is in `work/edited/edit.safetensors` at the end is your submission. It may be empty if you find no edit you trust. Your final reply should cover:

1. how you approached the problem, step by step
2. the mechanism you found and the evidence for it
3. the edit: which tensors and how many entries
4. its measured effect on held-out practice problems, with sample sizes
5. how confident you are, and what you would do with more budget
