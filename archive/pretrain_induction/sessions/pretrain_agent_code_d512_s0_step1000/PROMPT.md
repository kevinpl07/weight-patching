# Task: give a small transformer's pretraining a head start by editing its weights

You are working on a small language model **in the middle of pretraining**. You will read its weights and activations and write new values into them directly. After your edit, pretraining resumes and runs to the end. Your goal is to make **the rest of that training reach lower loss, sooner**.

There is no training on your side: no gradients, no optimizer, and no fitting of weights by regression or by search against a loss. You may compute statistics from weights and activations (means, differences, projections, PCA/SVD, norms, attention patterns) and construct an edit from what you find. Comparing a handful of hand-chosen variants, for example to check dose–response, is fine. Automated search loops over edit parameters are not.

## The model and the run

- **Architecture:** a 2-layer **attention-only** transformer: no MLPs. It has learned token and position embeddings (`tok.weight`, `pos.weight`), 8 heads of size 64 per layer, width 512 and a 512-token context. Each block applies pre-LayerNorm (`blocks.{l}.ln1`) and then attention, with fused `blocks.{l}.attn.qkv.weight` of shape [1536, 512]: rows 0–511 are Q, 512–1023 are K, 1024–1535 are V, and inside each, head h is rows h·64…h·64+63. The output is `blocks.{l}.attn.proj.weight` [512, 512], and head h is columns h·64…h·64+63. A final LayerNorm (`ln_f`) feeds an untied unembedding (`head.weight`). There are no biases in the linear layers. The exact code is in `pretrain/model.py`.
- **Data:** Python source files, byte-level BPE with 8,192 tokens, files joined by `<|eos|>` (id 0). Training windows are random 512-token slices.
- **Training recipe:** AdamW with betas (0.9, 0.95), weight decay 0.1 on matrices and gradient clipping at 1.0. The learning rate peaks at 1e-3 after a 100-step warmup, then follows a cosine decay to 1e-4 at step 15,000. Each step uses 64 sequences of 512 tokens.
- **Where you are:** the checkpoint is at **step 1,000 of 15,000**. `loss_so_far.csv` has the training loss up to here. After your edit, training resumes from step 1,000 with the **same optimizer state and the same sequence of batches**, and runs to step 15,000.

The edit does not need to survive training unchanged. Training will keep updating every weight. What counts is how the loss of the continued run compares with continuing from the unedited checkpoint.

## How you will be scored (you will not see this data)

- **Held-out validation loss** of the continued run at every point from step 1,000 to 15,000. The main number is the average over that span. Also reported: the loss at steps 1,500, 2,000, 3,000, 5,000 and 15,000.
- **Controls:**
  - the same run continued from the unedited checkpoint
  - random edits that change the same tensors by the same amount: the same number of entries and the same norm of change per tensor

  Your edit only counts if it beats both.

## Tools

The helpers are in `pretrain/agent_tools.py` (read it, but don't modify it). Run every script from this folder:

```
cd <project>/runs/pretrain_agent/code_d512_s0_step1000
PYTHONIOENCODING=utf-8 AGENT_SESSION=. ../../../.venv/Scripts/python.exe your_script.py
```

Start each script with:

```
import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
```

- **Free:**
  - `T.info()`: config, parameter shapes, recipe
  - `T.base_weights()`: the checkpoint as float32 CPU tensors
  - `T.tokenizer()`
  - `T.train_windows(n, length, seed)`: training-split text
  - `T.current_edit()`
  - `T.save_edit({name: full_new_tensor}, note=...)`
  - `T.budget()`
- **Counted forward calls (budget 1,000; each takes milliseconds):**
  - `T.run(tokens, edited=..., attn=..., resid=...)`: per-position loss, attention patterns, the residual stream, and each head's output
  - `T.metrics(edited=...)`: loss on fixed training windows, loss by position, a copy-benefit score, and per-head previous-token and induction-style attention scores. Read its docstring.
- The training split is yours to use freely. The held-out validation data used for scoring is not available to you.

## Rules

- **Edit budget:** at most **400,000** weight entries may differ from the checkpoint, about 3.7% of the model's 10.8M. Smaller is fine.
- **Out of scope:** `tok.weight` and `head.weight`. You may edit `blocks.*`, `pos.weight` and `ln_f.*`.
- **No training:** no gradients or autograd, no optimizer, and no fitting weights by regression or search against a loss (see above).
- **Show your evidence.** Explain what the model is missing or doing badly at step 1,000, why your edit should help the training that follows, and what the edit does right away: its effect on training-split loss, by position, and on the diagnostics. Check dose–response.
- **Stay in your lane.** Work only in this folder, and read only `pretrain/agent_tools.py` and `pretrain/model.py` outside it. Don't open other `runs/` folders, logs, checkpoints or evaluation scripts, and don't use the internet.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. Keep copies of promising edits. The current edit lives in `work/edited/edit.pt`. Don't print huge tensors.
- **Time:** aim to finish within about 60–90 minutes and about 120 tool calls.

## Submission

Whatever is in `work/edited/edit.pt` at the end is your submission. It may be empty if you find no edit you trust. Your final reply should cover:

1. how you approached the problem, step by step
2. what you found missing or broken at step 1,000, and the evidence
3. the edit: which tensors, how many entries, and the norm of the change
4. its immediate measured effect on training-split loss and diagnostics
5. your prediction for the continued training (does the advantage grow, persist or fade?), and how confident you are
