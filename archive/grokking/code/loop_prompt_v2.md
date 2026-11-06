# Mid-training weight edit

A small neural network is in the middle of its training run. Training is paused at step {STEP}. This is edit round {ROUND} of {N_ROUNDS}: rounds happen at pauses {ROUND_STEPS} steps apart, each in a separate session with these same instructions, and training continues after each edit. Read `history.md` first. You can read its weights and write new values into them directly. Once you submit, training resumes from your edited weights with the same optimizer state and settings. {AFTER}

**Goal:** edit the weights so that, once training resumes, the network reaches high accuracy on held-out pairs in as few steps as possible, and stays there until the run ends at step {END}.

**Your edit is a starting point for the optimizer, not the finished network.** It is judged by where training goes after it: held-out accuracy over the rest of the run and at the end. It is not judged by the network right after the edit. The optimizer keeps running after you submit. Changes it disagrees with tend to be undone, and anything you remove, it may be unable to bring back. Prefer edits the optimizer will build on.

A useful way to think about the job: find what is slowing the optimizer down, and remove that.

## The network
- Task: predict (a + b) mod {P} for integers 0 ≤ a, b < {P}. The input is concat(one_hot(a), one_hot(b)), which has 2×{P} dimensions. The output is {P} logits.
- Model: fc1 Linear(2×{P}, {HIDDEN}) → ReLU → fc2 Linear({HIDDEN}, {P}). {TOTAL} weights in total.
- Training: full-batch AdamW, lr {LR}, weight decay {WEIGHT_DECAY}, betas ({BETA1}, {BETA2}), cross-entropy on {N_TRAIN} training pairs.
- State at the pause: training accuracy {TRAIN_ACC}, held-out accuracy {TEST_ACC}.

## Rules
These rules are the design of the experiment. Keep to them even if breaking one looks helpful.
1. **No optimizer.** Change weights only by writing values with the `edit` tool. Do not use gradient descent, train or fine-tune anything, or fit weights to data (least squares, regression and the like). Do not run an automated search that scores many candidate edits and keeps the best. You may write code to analyze the weights, and to compute the values you write from your own reasoning, for example from a formula you chose. Comparing a few hand-designed candidates on the training set is fine. You may read gradients and optimizer state with `optim`, and watch short throwaway runs with `trial`. Do not write values made by stepping along a gradient (w − η·g or similar).
2. **Training pairs only.** Every forward pass you run uses training pairs (`train_pairs.npy`, or the `probe` tool). Do not build, evaluate on or reason from held-out pairs. The only held-out numbers you may use are the ones `history.md`, `meta.json` and the `trial` tool report.
3. **Budget.** At most {BUDGET} weight entries may differ from the paused checkpoint. The `edit` tool enforces this. Optimizer-state resets don't count toward the budget.
4. **Stay in your session.** Work only in `{SESSION}`. Save every analysis script you write to `{SESSION}/scratch/`. Apart from your session, read only `edit_tools.py` and `grok.py` in `{ROOT}`.
5. **Submit exactly once,** with your predictions.

## Files in your session
- `weights.npz` holds the paused checkpoint and is read-only. `working.npz` holds your current edited weights. Both are numpy archives with keys fc1.weight, fc1.bias, fc2.weight and fc2.bias, in PyTorch layout (out_features, in_features).
- `train_pairs.npy` has columns a, b, label.
- `meta.json` holds the config and training history.
- `history.md` lists the earlier rounds: what each one changed, its submission note, the training and held-out accuracy since each edit, and each round's predictions next to what actually happened.
- `optimizer.npz` holds the AdamW state at the pause: `exp_avg/<tensor>`, `exp_avg_sq/<tensor>` and `step/<tensor>`.

## Tools
Run them with `"{ROOT}\.venv\Scripts\python.exe" "{ROOT}\edit_tools.py" "{SESSION}" <command>`:
- `info`: task, shapes, budget, training history
- `stats`: per-tensor stats, plus a per-neuron table (input norms for a and b, bias, output norm, how often each neuron is active)
- `show TENSOR --rows i:j --cols i:j`: print raw values
- `probe --n N [--hidden]` or `probe --pairs "a,b a,b"`: predictions on training pairs, with optional top hidden activations
- `optim [--neurons] [--save FILE]`: the full-batch training gradient at your working weights, the AdamW moments, and the update the optimizer will make on its next step, per tensor and, with `--neurons`, per neuron. `--save` writes them all to an .npz for your own analysis.
- `trial --steps N`: trains a throwaway copy from your current working weights for N ≤ 1,000 steps, with the real optimizer state and settings. It prints training loss, training accuracy and held-out accuracy every 100 steps. At most 3 trials per round; the copy is discarded.
- `edit FILE.json`: apply a list of ops to the working weights. Each op is `{"tensor": "fc1.weight", "rows": [...]|null, "cols": [...]|null, "op": "set"|"add"|"scale", "value": number or nested list}`. The op `"reset_optimizer"` (no value) zeroes the AdamW moments of the selected entries when training resumes. It applies only to entries you changed. See the docstring of edit_tools.py for details.
- `diff`, `reset`, and `submit --note "..." --predict-next X --predict-end Y`. X is the held-out accuracy you expect {ROUND_STEPS} steps after your edit, and Y is the held-out accuracy you expect at the end of the run (step {END}), both as fractions between 0 and 1. Your predictions are shown to the next rounds next to what actually happened.

Use the same Python interpreter for your own analysis scripts.

## Report
Your final message has four parts, under 300 words in total:
1. what you found in the weights
2. exactly what you changed, and why
3. what you expect to happen once training resumes
4. your confidence in that
