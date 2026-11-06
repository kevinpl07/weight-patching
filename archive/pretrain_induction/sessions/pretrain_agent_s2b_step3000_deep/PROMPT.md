# Task: set this network up so that the rest of its pretraining ends at a better model

You are working on a small language model **in the middle of pretraining**, at step 3,000 of 15,000. You will read its weights and activations, write new values into them, and then pretraining resumes and runs to step 15,000.

**The question:** is there a network with the same architecture and the same remaining training budget that ends up **better at step 15,000** than where this run is heading? Can you put this one onto that path?

Your edit is judged by the **final loss at step 15,000**, not by its effect right now. An edit that makes the loss **worse immediately is completely acceptable** if it sets up better learning for the remaining 12,000 steps. Don't optimize for the next 100 steps.

## History of this run

Read `previous_notes_1.md` and `previous_notes_2.md` first. Both were written by you in earlier sessions.

- **Step 1,000:** you planted a previous-token head (L0H6) and an induction head (L1H2), plus sinusoidal position features in `pos.weight`. The run pulled about 0.3 nats ahead of an unedited copy. The unedited copy then formed its own induction circuit around step 2,000, and the gap shrank to about 0.03 by step 3,000.
- **First session at step 3,000:** you concluded there was nothing to gain, because every immediate improvement was capped at about 0.015 nats, below the cost of repurposing a head. **That analysis looked at the wrong objective,** the loss right now. This session is a second attempt at the same checkpoint with the right objective: where the run ends up.

So far, the run with your step-1,000 edit and the unedited copy end at almost the same validation loss at step 15,000 (1.905 vs 1.911). Training erases a one-off head start. To matter at step 15,000, an edit has to change **what training can reach**, not just where it starts.

## Dig in

Think about the learning dynamics of the remaining 12,000 steps. What will AdamW do with this network, what limits how far it gets, and what structure would let gradient descent make more progress than it otherwise would? Some questions worth investigating (not a checklist, and not hints about the answer):

- **Capacity use:** which of the 16 heads are redundant, duplicated, near-dead or doing something of low value? Would training use them better if they were freed up, re-initialized, or given a different starting structure?
- **Composition:** what can layer-1 heads read from layer 0's output? Is there information layer 1 could use if layer 0 wrote it, such as more previous-token or position features, multi-token match information, or a richer token code? What useful two-step (Q-, K- or V-composition) mechanisms are missing?
- **What is still wrong:** what kinds of tokens and contexts carry the most loss? What patterns in Python, such as brackets, indentation, scoping, names, strings or repetition at other offsets, would need structure the model doesn't have?
- **Optimization health:** are there weight norms, LayerNorm scales, attention logit scales or saturated softmaxes that slow learning, for example heads whose attention is so sharp or so flat that their gradients vanish? How large are Adam's updates relative to the weights at this learning rate?
- **Position code:** is the positional representation good enough for what the remaining heads would need?

Form hypotheses, test them with the tools, and build the edit you believe in most.

## The model and the run

- **Architecture:** a 2-layer **attention-only** transformer: no MLPs. It has learned `tok.weight` and `pos.weight`, 8 heads of size 64 per layer, width 512 and a 512-token context. Each block applies pre-LayerNorm (`blocks.{l}.ln1`) and then attention, with fused `blocks.{l}.attn.qkv.weight` of shape [1536, 512]: rows 0–511 are Q, 512–1023 are K, 1024–1535 are V, and inside each, head h is rows h·64…h·64+63. The output is `blocks.{l}.attn.proj.weight` [512, 512], and head h is columns h·64…h·64+63. A final `ln_f` feeds an untied unembedding (`head.weight`). There are no linear biases. The code is in `pretrain/model.py`.
- **Data:** Python source files, byte-level BPE with 8,192 tokens, files joined by `<|eos|>` (id 0). Training windows are random 512-token slices.
- **Recipe:** AdamW with betas (0.9, 0.95), weight decay 0.1 on matrices and gradient clipping at 1.0. The learning rate peaks at 1e-3 after a 100-step warmup, then follows a cosine decay to 1e-4 at step 15,000; at step 3,000 it is about 9.2e-4. Each step uses 64 × 512 tokens.
- **Continuation:** after your edit, training resumes from step 3,000 with the **same optimizer state** and the real batch sequence, until step 15,000.

## How you will be scored (you will not see this data)

- **Main number:** held-out validation loss at step 15,000.
- **Also reported:** the average over steps 3,000–15,000.
- **Compared against:**
  - continuing from this checkpoint without your new edit
  - random edits of the same size, same tensors and same norm of change per tensor

## Tools

Use `pretrain/agent_tools.py` (read it, but don't modify it). Run scripts from this folder:

```
cd <project>/runs/pretrain_agent/code_d512_s0_multi/s2b_step3000_deep
PYTHONIOENCODING=utf-8 AGENT_SESSION=. AGENT_LOOKAHEADS=10 ../../../../.venv/Scripts/python.exe your_script.py
```

Start each script with:

```
import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
```

- **Free:**
  - `T.info()`
  - `T.base_weights()`
  - `T.tokenizer()`
  - `T.train_windows(n, length, seed)`
  - `T.current_edit()`
  - `T.save_edit(...)`
  - `T.budget()`
- **Forward calls (budget 1,000):**
  - `T.run(tokens, edited=..., attn=..., resid=...)`
  - `T.metrics(edited=...)`
- **New:** `T.lookahead(steps=..., edited=True/False, every=100, label=...)` (budget **10 runs**, at most 1,500 steps each, about 1.5 minutes per 1,500 steps):
  - It continues *training* from this checkpoint with the real recipe and optimizer state, either with your current edit applied or without it, and reports the training-window loss and copy benefit along the way.
  - It uses a different batch order from the real continuation, but every lookahead run uses the **same** batches. So an edited run and a no-edit run of the same length are a paired comparison.
  - Run-to-run differences under about 0.005 nats are noise.
  - **Use it to verify how a hand-built edit trains, not to search.** You may compare a handful of hand-designed variants. Automated sweeps are not allowed.
  - Remember that 1,500 steps is 12% of the remaining run. An edit can trail at step 4,500 and still win by 15,000, or the reverse. Reason about the trend, not only the endpoint.

## Rules

- **Edit budget:** at most **400,000** entries may differ from this checkpoint. **Out of scope:** `tok.weight` and `head.weight`.
- **No gradients or optimizer of your own,** and no fitting weights by regression or by automated search against a loss. You build the edit from your analysis; `lookahead` is only for checking it.
- **Stay in this folder,** and read only `pretrain/agent_tools.py` and `pretrain/model.py` outside it. Don't open other `runs/` folders, logs or checkpoints, and don't use the internet.
- **Keep `work/notes.md` up to date:** hypotheses, tests, conclusions.
- **Time:** up to about 2 hours and about 200 tool calls. Use them. An empty edit is still allowed, but only after a serious search of the ideas above, backed by evidence.

## Submission

Whatever is in `work/edited/edit.pt` at the end is your submission. Your final reply should cover:

1. the hypotheses you investigated and what the evidence said for each
2. the edit: which tensors, how many entries, the norm of the change, and the mechanism it is meant to create or enable
3. its immediate effect, and its lookahead trajectory against no edit (paired)
4. your prediction for the loss at step 15,000 relative to no edit, with confidence and reasoning
