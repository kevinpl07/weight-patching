# Task: keep improving a small transformer's pretraining by editing its weights (intervention {k} of {n_total})

You are working on a small language model **in the middle of pretraining**. You will read its weights and activations and write new values into them directly. After your edit, pretraining resumes. Your goal is to make **the rest of that training reach lower loss, sooner**.

This run is being edited repeatedly. {history} Your notes from the earlier session(s) are in `previous_notes_*.md` in this folder. Read them first. Since the last edit, training has kept updating every weight, so don't assume your earlier construction is still intact. Check what the model looks like **now**.

There is no training on your side: no gradients, no optimizer, and no fitting of weights by regression or by search against a loss. You may compute statistics from weights and activations (means, differences, projections, PCA/SVD, norms, attention patterns) and construct an edit from what you find. Comparing a handful of hand-chosen variants, for example to check dose–response, is fine. Automated search loops over edit parameters are not.

## The model and the run

- **Architecture:** a 2-layer **attention-only** transformer: no MLPs. It has learned token and position embeddings (`tok.weight`, `pos.weight`), 8 heads of size 64 per layer, width 512 and a 512-token context. Each block applies pre-LayerNorm (`blocks.{{l}}.ln1`) and then attention, with fused `blocks.{{l}}.attn.qkv.weight` of shape [1536, 512]: rows 0–511 are Q, 512–1023 are K, 1024–1535 are V, and inside each, head h is rows h·64…h·64+63. The output is `blocks.{{l}}.attn.proj.weight` [512, 512], and head h is columns h·64…h·64+63. A final LayerNorm (`ln_f`) feeds an untied unembedding (`head.weight`). There are no biases in the linear layers. The exact code is in `pretrain/model.py`.
- **Data:** Python source files, byte-level BPE with 8,192 tokens, files joined by `<|eos|>` (id 0). Training windows are random 512-token slices.
- **Training recipe:** AdamW with betas (0.9, 0.95), weight decay 0.1 on matrices and gradient clipping at 1.0. The learning rate peaks at 1e-3 after a 100-step warmup, then follows a cosine decay to 1e-4 at step 15,000. Each step uses 64 sequences of 512 tokens.
- **Where you are:** the checkpoint is at **step {step} of 15,000**. `loss_so_far.csv` has the training loss from step 0 to here, including the effect of earlier edits. After your edit, training resumes from step {step} with the **same optimizer state and the same sequence of batches**. {next_stop}

What counts is how the continued run's loss compares with continuing from this checkpoint **without your new edit**.

## How you will be scored (you will not see this data)

- **Held-out validation loss** of the continued run from step {step} onward. The main number is the average over that span; the loss at several later steps is also reported.
- **Controls:**
  - the same run continued from this checkpoint without your new edit
  - random edits that change the same tensors by the same amount: the same number of entries and the same norm of change per tensor

  Your edit only counts if it beats both. An empty edit is a legitimate submission if you find nothing that would help: a harmful edit scores worse than no edit.

## Tools

The helpers are in `pretrain/agent_tools.py` (read it, but don't modify it). Run every script from this folder:

```
cd {session_abs}
PYTHONIOENCODING=utf-8 AGENT_SESSION=. {python_rel} your_script.py
```

Start each script with:

```
import sys; sys.path.insert(0, "{tools_rel}"); import agent_tools as T
```

- **Free:**
  - `T.info()`
  - `T.base_weights()`: this checkpoint as float32 CPU tensors
  - `T.tokenizer()`
  - `T.train_windows(n, length, seed)`: training-split text
  - `T.current_edit()`
  - `T.save_edit({{name: full_new_tensor}}, note=...)`
  - `T.budget()`
- **Counted forward calls (budget 1,000; each takes milliseconds):**
  - `T.run(tokens, edited=..., attn=..., resid=...)`: per-position loss, attention patterns, the residual stream, and each head's output
  - `T.metrics(edited=...)`: loss on fixed training windows, loss by position, a copy-benefit score, and per-head previous-token and induction-style attention scores
- The training split is yours to use freely. The held-out validation data used for scoring is not available to you.

## Rules

- **Edit budget:** at most **400,000** weight entries may differ from **this** checkpoint, about 3.7% of the model's 10.8M. Smaller is fine.
- **Out of scope:** `tok.weight` and `head.weight`. You may edit `blocks.*`, `pos.weight` and `ln_f.*`.
- **No training:** no gradients or autograd, no optimizer, and no fitting weights by regression or search against a loss.
- **Show your evidence.** Explain what the model is missing or doing badly at step {step}, why your edit should help the training that follows, and what the edit does right away: its effect on training-split loss, by position, and on the diagnostics. Check dose–response.
- **Stay in your lane.** Work only in this folder, and read only `pretrain/agent_tools.py` and `pretrain/model.py` outside it. Don't open other `runs/` folders, logs, checkpoints or evaluation scripts, and don't use the internet.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. The current edit lives in `work/edited/edit.pt`. Don't print huge tensors.
- **Time:** aim to finish within about 60 minutes and about 100 tool calls.

## Submission

Whatever is in `work/edited/edit.pt` at the end is your submission. It may be empty. Your final reply should cover:

1. how you approached the problem, step by step
2. what you found missing or improvable at step {step}, and the evidence
3. the edit: which tensors, how many entries, and the norm of the change
4. its immediate measured effect on training-split loss and diagnostics
5. your prediction for the continued training (does the advantage grow, persist or fade?), and how confident you are
