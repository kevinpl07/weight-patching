# Task: repair a pretrained image classifier's spurious features by editing its weights

You are working on a **pretrained ImageNet classifier** that relies on **spurious features** for some classes. Given a photo that shows only a class's typical context or a co-occurring object, and no object of the class itself, it still predicts the class or gives it high probability. You will read its weights and activations and write new values into them directly. **Your edit is final.** No training happens afterwards, and the edited network is what gets scored.

There is no training on your side: no gradients, no optimizer, and no fitting of weights by regression or by search against a loss. You may compute statistics from weights and activations (means, differences, projections, PCA/SVD, norms, per-channel activation profiles) and construct an edit from what you find. Comparing a handful of hand-chosen variants, for example to check dose–response, is fine. Automated search loops over edit parameters are not.

## The model

- **Architecture:** torchvision `resnet50` with the original `IMAGENET1K_V1` weights (76.1% top-1 on ImageNet val), 25.6M parameters, in **eval mode**, so BatchNorm uses its running statistics.
  - stem: `conv1`, `bn1`, ReLU and max-pool
  - four stages `layer1`–`layer4` of 3/4/6/3 Bottleneck blocks: `conv1`/`bn1` 1×1, `conv2`/`bn2` 3×3, `conv3`/`bn3` 1×1, plus `downsample.0`/`.1` in each stage's first block; the block output is `relu(bn3(conv3(...)) + identity)`
  - global average pooling (`avgpool`) to a 2,048-d nonnegative feature
  - `fc` (`fc.weight` [1000, 2048], `fc.bias` [1000])
- **Inputs:** 224×224 RGB, normalized with the ImageNet mean and std (the tools do this for you).

## The failure you are asked to fix (your dev data)

`T.dev()` gives you 4,250 preprocessed images:
- **1,250 spurious images:** 25 for each of 50 classes (the "dev classes"). Each image is labelled with the class k it is wrongly pushed towards. It shows only k's spurious cue and **contains no object of class k**. Many of these images are classified as k today. Look at some of them (`T.show`).
- **3,000 clean images:** real photos from ImageNetV2, 3 per class for all 1,000 classes, labelled with their true class.

You are not told which cue each class depends on; finding it is part of the task. You may inspect any image, activation or weight in the dev data.

## How you will be scored (you will not see this data)

- **Spurious reliance, for the 50 dev classes:** new, held-out spurious images of the same 50 classes (50 per class). Two numbers:
  - **hit rate:** how often the image is predicted as k
  - **spurious AP:** per class, the average precision of the softmax probability p_k when ranking held-out clean photos of class k above the spurious images of k (higher is better)
- **Collateral damage:** top-1 accuracy on 7,000 held-out ImageNetV2 photos of all 1,000 classes, overall and on the 50 dev classes. Losing clean accuracy, including on the dev classes themselves, counts against you.
- Also reported: other spurious classes you were not shown (no one expects you to fix those), and the number of entries you changed.
- **Controls:**
  - random edits that change the same entries by the same amount
  - a simple scripted projection edit built from the same dev data
  - retraining the last layer on the same dev data

  The question is whether careful reading and editing of the weights repairs the failure without training and without collateral damage.

## Tools

The helpers are in `vision_spurious/agent_tools.py` (read it, but don't modify it). Run every script from this folder:

```
cd <project>/runs/vision_spurious/session_1
PYTHONIOENCODING=utf-8 AGENT_SESSION=. ../../../.venv/Scripts/python.exe your_script.py
```

Start each script with:

```
import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
```

- **Free:**
  - `T.info()`: model summary, budgets, the dev classes and their names
  - `T.base_weights()`: all weights as float32 CPU tensors
  - `T.dev()`: dev images and labels
  - `T.categories()`: the 1,000 class names
  - `T.show(indices, "work/name.png")`: saves a labelled image grid; open it with your file-reading tool to look at it
  - `T.current_edit()`
  - `T.save_edit({name: full_new_tensor}, note=...)`
  - `T.budget()`
- **Counted forward passes (budget 300,000 images; at most 64 images per call; a full dev pass takes about 15–30 s):**
  - `T.run(images, edited=..., layers=[...])`: logits, probabilities, and the outputs of any named modules (e.g. `"avgpool"` for the 2,048-d penultimate feature, `"layer4"`, `"layer4.2.conv3"`)
  - `T.metrics(edited=...)`: dev-split diagnostics: clean top-1, spurious hit rate, mean p_k and AP per dev class. It costs 4,250 images; read its docstring.
- The GPU is shared with another job, so keep to the tools' batch sizes. Cache activations you reuse (for example the base model's penultimate features for all of dev, saved with `torch.save` in `work/`) instead of recomputing them.

## Rules

- **Edit budget:** at most **200,000** weight entries may differ from the original, about 0.78% of the model's 25.6M. Any float tensor in the state dict may be edited (conv and fc weights, biases, BN weights, biases and running statistics). Smaller is fine.
- **No training:** no gradients or autograd, no optimizer, and no fitting weights by regression or search against a loss (see above).
- **Show your evidence.** Explain:
  - which cue drives each class you fix (or each group of classes) and where it lives in the network
  - why your edit removes it
  - what the edit does right away on dev: the spurious hit rate and AP, and clean accuracy overall and on the dev classes

  Check dose–response. Remember the dev set is small: 3 clean photos per class. Don't overfit it, and think about what clean images of other classes might share the cue you remove.
- **Stay in your lane.** Work only in this folder, and read only `vision_spurious/agent_tools.py` and `vision_spurious/common.py` outside it. Don't open other `runs/` folders, `data_vision/derived/eval.pt`, evaluation or baseline scripts, or other files in the repository, and don't use the internet.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. Keep copies of promising edits. The current edit lives in `work/edited/edit.pt`. Don't print huge tensors.
- **Time:** you have about **30 minutes** and about 60 tool calls. Make sure a sensible edit is saved well before the end.

## Submission

Whatever is in `work/edited/edit.pt` at the end is your submission. It may be empty if you find no edit you trust. Your final reply should cover:

1. how you approached the problem, step by step
2. what drives the spurious predictions, with the evidence
3. the edit: which tensors, how many entries, and the norm of the change
4. its measured effect on dev: spurious hit rate and AP, and clean top-1 overall and on the dev classes
5. your prediction for the held-out scores (hit rate, AP, overall top-1 change), and how confident you are
