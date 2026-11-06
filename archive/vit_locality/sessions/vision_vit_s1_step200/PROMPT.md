# Task: give a small vision transformer's training a lasting boost by editing its weights

You are working on a small vision transformer (ViT) **in the middle of training** on CIFAR-10. You will read its weights and activations and write new values into them directly. After your edit, training resumes and runs to the end. Your goal is to make **the rest of that training reach higher held-out accuracy: above all at the end of training, and also sooner along the way**.

There is no training on your side: no gradients, no optimizer, and no fitting of weights by regression or by search against a loss. You may compute statistics from weights and activations (means, differences, projections, PCA/SVD, norms, attention patterns) and construct an edit from what you find, including structure you write down from your own understanding. Comparing a handful of hand-chosen variants, for example to check dose–response, is fine. Automated search loops over edit parameters are not.

## The model and the run

- **Architecture:** a ViT with patch size 4 on 32×32 images, so each image is an **8×8 grid of 64 patch tokens**; token t sits at grid row t // 8, column t % 8. There is no class token: the final LayerNorm output is averaged over the 64 tokens and fed to a linear classifier.
  - `patch_embed.weight` [192, 3, 4, 4] and `patch_embed.bias`: a stride-4 convolution that embeds each patch.
  - `pos_embed` [1, 64, 192]: learned absolute position embeddings, added to the patch embeddings.
  - 6 pre-LayerNorm blocks. Block l applies `blocks.{l}.ln1`, then attention with a fused `blocks.{l}.attn.qkv.weight` [576, 192] and `qkv.bias` [576]: rows 0–191 are Q, 192–383 are K, 384–575 are V, and inside each, head h is rows h·64…h·64+63. There are 3 heads of size 64, and the attention is bidirectional (no mask), with scores q·k/√64. The output projection is `blocks.{l}.attn.proj.weight` [192, 192] (head h is columns h·64…h·64+63) plus `proj.bias`. Then `blocks.{l}.ln2` and an MLP `blocks.{l}.mlp.fc1` [768, 192] → GELU → `blocks.{l}.mlp.fc2` [192, 768], both with biases. Both sublayers are residual.
  - A final LayerNorm `norm`, mean-pooling over tokens, and the classifier `head.weight` [10, 192], `head.bias`.
  - 2,693,194 parameters in total. The exact code is in `vision/model.py`.
- **Data:** the CIFAR-10 training set (50,000 32×32 colour images, 10 classes, 5,000 per class), normalized per channel. Augmentation: a random 32×32 crop of the image zero-padded by 4 pixels, and a random horizontal flip. Nothing else: no label smoothing, mixup, dropout or stochastic depth.
- **Training recipe:** AdamW with betas (0.9, 0.999), weight decay 0.05 on weight matrices and the patch-embedding kernel (not on biases, LayerNorms or `pos_embed`), and gradient-norm clipping at 1.0, in bf16 autocast. The learning rate peaks at 0.001 after a 200-step linear warmup, then follows a cosine decay to 1e-05 at step 4,000. Each step uses 512 images (97 steps per epoch), so the whole run is about 41 epochs. The loss is plain cross-entropy.
- **Where you are:** the checkpoint is at **step 200 of 4,000** (about 2.1 epochs in). `loss_so_far.csv` has the training loss and training-batch accuracy up to here (augmented batches, means over 10 steps). After your edit, training resumes from step 200 with the **same optimizer state and the same sequence of batches and augmentation draws**, and runs to step 4,000.

The edit does not need to survive training unchanged. Training will keep updating every weight. What counts is how the accuracy of the continued run compares with continuing from the unedited checkpoint.

## How you will be scored (you will not see this data)

- **Held-out test accuracy** (the 10,000 CIFAR-10 test images) of the continued run, measured every 100 steps from step 200 to 4,000. The main number is the **final test accuracy at step 4,000**. Also reported: the average test accuracy over the span, and how many steps the run needs to reach fixed accuracy levels.
- **Controls:**
  - the same run continued from the unedited checkpoint
  - random edits that change the same tensors by the same amount: the same number of entries and the same norm of change per tensor
  - a scripted reference intervention from the literature, applied at the same step

  Your edit only counts if it beats the first two. The third is a reference point.

## Tools

The helpers are in `vision/agent_tools.py` (read it, but don't modify it). Run every script from this folder:

```
cd <project>/runs/vision_vit/s1_step200
PYTHONIOENCODING=utf-8 AGENT_SESSION=. ../../../.venv/Scripts/python.exe your_script.py
```

Start each script with:

```
import sys; sys.path.insert(0, "../../../vision"); import agent_tools as T
```

- **Free:**
  - `T.info()`: config, parameter shapes, recipe, class names
  - `T.base_weights()`: the checkpoint as float32 CPU tensors
  - `T.train_set()`: the whole training set as uint8 images [50000, 32, 32, 3] and labels
  - `T.train_images(n, seed, augment_like_training=False)`: n training images, normalized as in training, and their labels
  - `T.current_edit()`
  - `T.save_edit({name: full_new_tensor}, note=...)`
  - `T.budget()`
- **Counted forward calls (budget 1,000; each takes milliseconds):**
  - `T.run(images, labels, edited=..., attn=..., resid=...)`: logits, per-image loss, attention patterns of every head, the residual stream, and each head's and each MLP's output
  - `T.metrics(edited=...)`: loss and accuracy on fixed training images (also under one draw of the training augmentation), per-class accuracy, and per-head attention statistics (mean attention distance on the 8×8 grid, mass on the 3×3 neighbourhood, mass on the token itself). Read its docstring.
- The training set is yours to use freely. The test set used for scoring is not available to you. The GPU may be shared with other jobs, and each session is limited to a small slice of its memory; keep batches at or below 1,024 images.

## Rules

- **Edit budget:** at most **900,000** weight entries may differ from the checkpoint, about 33.4% of the model's 2,693,194. That is enough to rewrite every attention matrix in every block. Smaller is fine.
- **Out of scope:** `patch_embed.*` and `head.*`. You may edit `blocks.*`, `pos_embed` and `norm.*`.
- **No training:** no gradients or autograd, no optimizer, and no fitting weights by regression or search against a loss (see above).
- **Show your evidence.** Explain what the model is missing or doing badly at step 200, why your edit should help the training that follows, and what the edit does right away: its effect on training-set loss and accuracy, and on the diagnostics. Check dose–response.
- **Stay in your lane.** Work only in this folder, and read only `vision/agent_tools.py`, `vision/model.py` and `vision/data.py` outside it. Don't open other `runs/` folders, logs, checkpoints, data files or evaluation scripts, and don't use the internet. Get images only through the tools.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. Keep copies of promising edits. The current edit lives in `work/edited/edit.pt`. Don't print huge tensors.
- **Time:** you have **about 30 minutes in total** (about 50 tool calls). Stop and submit by then; an unfinished investigation with a careful small edit, or an empty edit, is better than running over.

## Submission

Whatever is in `work/edited/edit.pt` at the end is your submission. It may be empty if you find no edit you trust. Your final reply should cover:

1. how you approached the problem, step by step
2. what you found missing or broken at step 200, and the evidence
3. the edit: which tensors, how many entries, and the norm of the change
4. its immediate measured effect on training-set loss, accuracy and diagnostics
5. your prediction for the continued training (does the advantage grow, persist or fade, and what final test accuracy do you expect relative to the unedited run?), and how confident you are
