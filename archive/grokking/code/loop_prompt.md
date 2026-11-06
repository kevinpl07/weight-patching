# Mid-training weight edit

A small neural network is in the middle of its training run. Training is paused at step {STEP}. This is edit round {ROUND} of {N_ROUNDS}: earlier rounds edited the weights at earlier pauses, each in a separate session with these same instructions, and training continued after each edit. Read `history.md` first. You can read its weights and write new values into them directly. Once you submit, training resumes from your edited weights with the same optimizer state and settings. {AFTER}

**Goal:** edit the weights so that, once training resumes, the network reaches high accuracy on held-out pairs in as few steps as possible.

## The network
- Task: predict (a + b) mod {P} for integers 0 ≤ a, b < {P}. The input is concat(one_hot(a), one_hot(b)), which has 2×{P} dimensions. The output is {P} logits.
- Model: fc1 Linear(2×{P}, {HIDDEN}) → ReLU → fc2 Linear({HIDDEN}, {P}). {TOTAL} weights in total.
- Training: full-batch AdamW, lr {LR}, weight decay {WEIGHT_DECAY}, betas ({BETA1}, {BETA2}), cross-entropy on {N_TRAIN} training pairs.
- State at the pause: training accuracy {TRAIN_ACC}, held-out accuracy {TEST_ACC}.

## Rules
These rules are the design of the experiment. Keep to them even if breaking one looks helpful.
1. **No optimizer.** Change weights only by writing values with the `edit` tool. Do not use gradient descent, train or fine-tune anything, or fit weights to data (least squares, regression and the like). Do not run an automated search that scores many candidate edits and keeps the best. You may write code to analyze the weights, and to compute the values you write from your own reasoning, for example from a formula you chose. Comparing a few hand-designed candidates on the training set is fine.
2. **Training pairs only.** Every forward pass uses training pairs (`train_pairs.npy`, or the `probe` tool). Do not build, evaluate on or reason from held-out pairs.
3. **Budget.** At most {BUDGET} weight entries may differ from the paused checkpoint. The `edit` tool enforces this.
4. **Stay in your session.** Work only in `{SESSION}`. Save every analysis script you write to `{SESSION}/scratch/`. Apart from your session, read only `edit_tools.py` and `grok.py` in `{ROOT}`.
5. **Submit exactly once.**

## Files in your session
- `weights.npz` holds the paused checkpoint and is read-only. `working.npz` holds your current edited weights. Both are numpy archives with keys fc1.weight, fc1.bias, fc2.weight and fc2.bias, in PyTorch layout (out_features, in_features).
- `train_pairs.npy` has columns a, b, label.
- `meta.json` holds the config and training history.
- `history.md` lists the earlier rounds: what each one changed, its submission note, and the training and held-out accuracy since each edit.

## Tools
Run them with `"{ROOT}\.venv\Scripts\python.exe" "{ROOT}\edit_tools.py" "{SESSION}" <command>`:
- `info`: task, shapes, budget, training history
- `stats`: per-tensor stats, plus a per-neuron table (input norms for a and b, bias, output norm, how often each neuron is active)
- `show TENSOR --rows i:j --cols i:j`: print raw values
- `probe --n N [--hidden]` or `probe --pairs "a,b a,b"`: predictions on training pairs, with optional top hidden activations
- `edit FILE.json`: apply a list of ops to the working weights. Each op is `{"tensor": "fc1.weight", "rows": [...]|null, "cols": [...]|null, "op": "set"|"add"|"scale", "value": number or nested list}`. See the docstring of edit_tools.py for details.
- `diff`, `reset`, and `submit --note "..."`

Use the same Python interpreter for your own analysis scripts.

## Report
Your final message has four parts, under 300 words in total:
1. what you found in the weights
2. exactly what you changed, and why
3. what you expect to happen once training resumes
4. your confidence in that
