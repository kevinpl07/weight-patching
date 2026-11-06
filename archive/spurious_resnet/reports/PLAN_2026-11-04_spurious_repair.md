# Plan (2026-11-04): can an agent's weight edit repair spurious features in a pretrained ImageNet classifier, with no training?

> **Outcome (2026-11-04): failed the pass bar, narrowly (0 of 3 sessions pass).** All three Opus edits (fc-only, 3k–30k entries) cut the held-out spurious hit rate from 50% to 16–22%. s2 and s3 met condition 1, s1 did not. Each session failed condition 2: s2 by 0.07 points overall (5 of 7,000 photos), s3 by 0.4 points on the dev classes (1–2 of 350 photos), and s1 by 2.2 points on the dev classes. Random controls did nothing. Every baseline (projection, SpuFix-style, last-layer retraining, layer4 fine-tune) failed condition 2 by far wider margins. Tables: [RESULTS.md](RESULTS.md); summary: [FINDINGS.md](FINDINGS.md). Post-hoc amendment A1 (below) adds a fair training counterfactual: training stopped at the agents' own operating points (chosen on dev) still has lower AP (0.47–0.56 against 0.57–0.67) and lower dev-class top-1 (46–55% against 57–60%), and no training checkpoint passes. A3: a closed-form script reproduces the agents' edits, and DFR-style balanced last-layer retraining, selected on dev only, passes the bar (hit 21.5%, AP 0.583, −0.29 overall, no dev-class loss). It is the only arm that does.

## Why this experiment

[FINDINGS.md](FINDINGS.md) ends with a clear prediction: weight edits should pay off where (1) there is a real failure with a visible symptom and (2) no training follows. Every earlier experiment had training after the edit, and training erased or made redundant whatever the edit did. The RL counterfactual added an open question: an edit changes very few weights, so it might avoid side effects that training causes.

This experiment is the Sparks setup in vision:
- a frozen, pretrained classifier
- a documented failure: spurious features
- the edit is final, so there's no optimizer afterwards

It also runs the "would training fix it" counterfactual on the same data, and measures collateral damage.

## Setup

- **Model:** torchvision ResNet-50, `IMAGENET1K_V1` weights (`resnet50-0676ba61.pth` from download.pytorch.org). It has 25.6M parameters and runs in eval mode, so BN uses running statistics.
- **Failure: Spurious ImageNet** (Neuhaus, Augustin, Boreiko and Hein, ICCV 2023; [YanNeu/spurious_imagenet](https://github.com/YanNeu/spurious_imagenet)). It has 100 classes × 75 real OpenImages photos. Each photo shows only a class's spurious cue and no object of the class, and is still pushed towards that class: for example, water without a boat.
- **Clean set: ImageNetV2 matched-frequency** (Hugging Face `vaishaal/ImageNetV2`: public, not gated, MIT licence). It has 10 images per class. Full ImageNet is not used anywhere.
- **Preprocessing:** the benchmark's own transforms, all bicubic.
  - ImageNetV2: resize to 256, then centre crop to 224.
  - Spurious images: resize to 224.

### Splits (seed 0, fixed by `vision_spurious/prepare_data.py` before any arm ran)

| Split | Who sees it | Contents |
|---|---|---|
| **dev** | agents, and every fitted or scripted baseline | ImageNetV2: 3 images per class (3,000). Spurious ImageNet: 50 "dev classes" × 25 images (1,250). |
| **eval** | only `eval.py` | ImageNetV2: the other 7 per class (7,000). Spurious: dev classes × 50 held-out images (2,500); 50 "unseen classes" × 75 (3,750). |

**What the agent is told, and why that's fair.** The agent is told that the model relies on spurious features that hurt some classes. It gets the dev spurious images labelled with the class they are wrongly pushed towards, as a user reporting a failure would provide. It doesn't get the other 50 benchmark classes or the benchmark's list of spurious components. The unseen-class eval group measures whether anything it does generalizes beyond the reported failures. Nobody is expected to fix the unseen classes, because their spurious cues differ from class to class.

### Metrics (all on eval)

**Spurious score.** We use the benchmark's own definition from its code. For each class k, it is the average precision (AP) of the softmax probability p_k, separating clean images of k (positives) from spurious images of k (negatives). Higher is better.
- **Simplification:** the official score uses 50 ImageNet-val positives per class. We have no ImageNet val, so we use the 7 ImageNetV2 eval images per class as positives. Absolute values are therefore not comparable with the paper's table; comparisons between arms are.
- Reported for the **dev classes** on their held-out images (primary) and for the **unseen classes**.
- **Also reported:**
  - ROC AUC
  - **spurious hit rate:** the fraction of spurious images predicted top-1 as k
  - mean p_k on spurious images

**Collateral damage.** ImageNetV2 eval top-1 (7,000 images), overall and split into dev classes (350 images), unseen spurious classes (350) and other classes (6,300).

**Edit size.** Entries changed.

## Arms

| Arm | What | Sees |
|---|---|---|
| **none** | base model | – |
| **agent s1–s3** | 3 independent blind Opus 5.5 sessions on the same model (the variance estimate); edit is final | weights + dev |
| **random s1–s3** | one per agent edit: the same tensors, the same entries, the same L2 norm of the change per tensor, with random Gaussian directions (`session.py random --mode same`). A second variant at random positions with the same count per tensor (`--mode count`), as in the earlier pilots, is also reported. | – |
| **proj** (scripted reference, weight edit) | for each dev class k: d_k = mean penultimate feature of k's dev spurious images − mean over all clean dev images (mu0). Remove fc row k's component along d_k, and adjust the bias so that mu0's logit is unchanged. Changes 50 × 2,049 = 102,450 entries. | dev |
| **spufix_like** (scripted reference, not a weight edit) | SpuFix's exact inference-time clamp formula (logit_k −= max(0, (w_k⊙phi − mu)·v)·sum(v)), with v_k = normalize(w_k ⊙ d_k) and mu = w_k ⊙ mu0. Real SpuFix needs ImageNet-train activations to match its neural PCA components, and the train set isn't available. So the direction comes from the dev data instead of neural PCA plus human labels. | dev |
| **llr** (last-layer retraining, the "would training fix it" counterfactual) | fc only, starting from the pretrained fc. Loss: CE on clean dev + mean −log(1 − p_k) on spurious dev. Full-batch Adam, lr 1e-4. It stops when the dev spurious hit rate is ≤ 2%, or at 1,000 steps. | dev |
| **ft_l4** (short fine-tune, the collateral-damage arm) | layer4 + fc (about 17M parameters), the same loss, Adam lr 1e-5, batch 64, BN frozen. It runs from cached layer3 outputs, so there is no augmentation. It stops at the end of the first epoch with dev hit rate ≤ 2%, or after 10 epochs. | dev |

Hyperparameters and stopping rules for llr and ft_l4 are fixed here. No tuning.

### The agent's rules (see `runs/vision_spurious/session_N/PROMPT.md`)

- **Access:** the weights and the dev split only. Tools are in `vision_spurious/agent_tools.py`:
  - forward passes, returning logits, probabilities, and the activations of any module
  - dev diagnostics
  - a viewer for dev images
  - `save_edit`, which checks the budget and scope
- **No training:** no gradients, no optimizer, no fitting by regression or search against a loss. Statistics and projections built by hand are fine, and so are a few hand-chosen dose checks.
- **Edit budget:** at most **200,000 entries (0.78% of the model)**, in any float tensor: conv, BN affine, BN running statistics, fc.
- **Forward budget:** 300,000 images (about 70 passes over dev).
- **Time:** about 30 minutes per session.
- **Objective given to the agent:** reduce the dev classes' reliance on their spurious cues, as measured on held-out images of those classes. Do this without hurting clean ImageNet accuracy on any class. It is told that held-out clean accuracy on all 1,000 classes is part of the score.

## Pass bar (fixed before any arm ran)

An agent session **passes** if all of the following hold on eval:
1. **Fixes the failure:** the dev-class spurious hit rate falls to at most half of the base model's, **and** the dev-class spurious AP rises by at least +0.10 absolute.
2. **No collateral damage:** overall ImageNetV2 top-1 drops by no more than 0.5 points (7,000 images, SE ≈ 0.6 points on a difference that pairs the same images, which is smaller), **and** top-1 on the dev classes drops by no more than 3 points.
3. **Not just any perturbation:** the matched random edit (`--mode same`) for that session does not meet condition 1.

**The experiment counts as a success** ("an agent can repair spurious features training-free") if **at least 2 of 3 sessions pass**.

### Reported comparisons (not pass/fail)
- **Agent vs proj:** does the agent beat a 20-line scripted edit that uses the same data? This is the real question about value added.
- **Agent vs llr and ft_l4:** does training fix it, and at what cost in collateral damage and entries changed? "The edit holds its own" if the agent's dev AP gain is within 0.05 of llr's and its overall top-1 drop is smaller than llr's.
- **Unseen classes:** any change, in either direction, for every arm.

## Predictions (written before running)

- **Base:** ImageNetV2 top-1 about 63%. Dev-class spurious hit rate about 30–45%, and AP about 0.4–0.55 with 7 positives.
- **proj:** cuts the dev-class hit rate by more than half, with less than 0.3 points of overall top-1 lost, and no change on unseen classes. Most of the cost falls on dev-class clean accuracy, by 1–3 points, because the spurious cue is real context for true class-k images too.
- **spufix_like:** similar to proj, slightly less collateral damage on clean images.
- **llr:** the largest dev-class fix (hit rate below 10%). Overall top-1 roughly unchanged, ±0.5, possibly a bit higher, because it also fits ImageNetV2's distribution. It changes all 2.05M fc entries.
- **ft_l4:** a dev fix similar to llr's, and the most collateral damage: overall top-1 falls by 1–3 points.
- **Agents:** they will work in the penultimate/fc space, finding channels that fire on the spurious images and not on clean class-k images, then shrinking those weights in fc row k. Some may edit layer4 channels. Expected: 2 of 3 sessions pass, a dev AP gain comparable to proj (±0.05), and no unseen-class effect. Random edits do nothing measurable.
- **Most likely headline:** the agent repairs the reported failures without training and with little collateral damage, but no better than a scripted projection. Retraining the last layer fixes it at least as well, at similar cost. So editing works here, and so does the cheap baseline.

## Files

- **Data:**
  - downloads, each in its own folder: `data_vision/resnet50_v1/`, `data_vision/imagenetv2_mf/`, `data_vision/spurious_imagenet/`
  - derived splits: `data_vision/derived/{dev,eval}.pt`
- **Code in `vision_spurious/`:**
  - `prepare_data.py`, `common.py`, `eval.py`
  - `agent_tools.py`, `session.py` (prepare / apply / random)
  - `baselines.py` (feats / proj / spufix / llr / ft)
- **Runs in `runs/vision_spurious/`:**
  - `eval/<arm>.json`
  - `arms/` (baseline weights and logs)
  - `session_{1,2,3}/` (agent sessions)

## Amendments

### Amendment A1, 2026-11-04 ~19:49: a fair training counterfactual (POST-HOC: written after all arms had been scored on eval)

**What was unfair.** The original llr and ft_l4 arms trained until the in-sample dev spurious hit rate was at most 2%. That pushed them to an extreme operating point:
- **The agents:** each compared several doses on dev and chose a moderate trade-off, with an in-sample dev hit rate of 0.234 (s1), 0.165 (s2) and 0.173 (s3).
- **The training arms:** they were never offered that choice.

So the original comparison answers "training until the symptom is gone", not "would plain training on the same images have done the agents' job".

**New arms** (`vision_spurious/train_curves.py`). All are trained on exactly the agents' dev data, starting from the pretrained weights, with the same loss as before (CE on clean dev + mean −log(1 − p_k) on spurious dev):

| Variant | Trains | Optimizer | Extra |
|---|---|---|---|
| `llr` | fc | full-batch Adam, lr 1e-4 | – |
| `llr_rows` | only the 50 dev-class rows of fc (weight + bias), the agents' scope | full-batch Adam, lr 1e-4 | – |
| `llr_gentle` | fc | Adam, lr 2e-5 | + KL(original ‖ new) on all 3,000 clean dev images |
| `ft` | layer4 + fc | Adam, lr 1e-5, batch 64 | – |
| `ft_gentle` | layer4 + fc | Adam, lr 1e-5, batch 64 | + KL on each batch's clean images |

**Checkpoint selection uses dev only.**
- **match_sN:** the first step at which the in-sample dev hit rate (computed as `T.metrics` does) falls to agent sN's in-sample dev hit rate.
- **cv:** a 3-fold check within dev. Each fold holds out 1 of the 3 clean photos per class and a third of each class's spurious images. The chosen step has the lowest CV hit rate among steps whose CV overall top-1 is no more than 0.5 points below CV base, and whose CV dev-class top-1 is no more than 3 points below. These are the pass bar's damage limits.

**Scoring.** Selected checkpoints are scored on eval with the original metrics and pass bar; the random-control condition (3) doesn't apply to training arms. Held-out scores along each trajectory are recorded only for a diagnostic trade-off plot (`runs/vision_spurious/tradeoff.png`) and never used for selection. Steps and training wall-clock to reach each matched point are reported.

**Status.** The original llr and ft_l4 numbers stay in the record, labelled as the "train to ≤2% hit" operating point.

### Amendment A2, 2026-11-04 ~20:10: training with a realistic amount of labelled data (POST-HOC design; written before its data was downloaded or any A2 arm ran)

**Why.** A1 trained on the agents' dev data only, which has 3 clean photos per class. Every earlier experiment's training had plenty of labelled data, so the fair question is whether training with a normal amount of clean data would match the edits.

**Data.** The user accepted the ImageNet terms for `ILSVRC/imagenet-1k`. We download about 12 train shards into `data_vision/imagenet_train_shards/`.
- `vision_spurious/imagenet_pool.py` builds a pool of up to 50 photos per class. It reads the shards with `python -I`, as data only, and preprocesses them like eval.
- The first min(10, n_k/5) photos of each class are held out as pool-val.

**Training** (`vision_spurious/train_full.py`).
- **Positives:** pool-train plus the 3 ImageNetV2 dev photos per class.
- **Negatives:** the same 25 dev cue-only images per dev class. No eval image is used.
- **Loss:** the A1 form. Each term is averaged over its own images, so positives and negatives carry equal total weight per step.
- **Variants:**
  - `ft_gentle_in` and `ft_in`: layer4 + fc, Adam lr 1e-5, 64 positives + 16 negatives per step, BN frozen; the first with KL to the original model on the positives.
  - `llr_gentle_in`: fc, full batch, lr 2e-5, + KL.
  - `llr_rows_in`: the 50 dev-class fc rows only, full batch, lr 1e-4.

**Stopping, without eval data:**
- **match_sN:** as in A1.
- **prac** (a practitioner rule):
  - Three runs, each trained on 2/3 of every class's cue images.
  - At each check, measure the CV cue hit rate on the held-out thirds, and pool-val top-1 overall and on the 50 dev classes.
  - Choose the step with the lowest CV hit rate among steps with pool-val top-1 no more than 0.5 points below base, and pool-val dev-class top-1 no more than 3 points below.
  - The full run, with all 25 cue images, is scored at that step.

**Scoring.** Same held-out metrics and pass bar.

**Caveat, decided in advance.** ResNet-50 was originally trained on these same ImageNet-train images. That is exactly the data a practitioner would have, so it is the right counterfactual, but the pool is not new to the model.

**Status (run 2026-11-04 ~20:45).** The user downloaded the 12 shards themselves (an earlier automated download had been blocked by the permission system). The pool has 48,092 photos, at least 23 per class and at least 39 per dev class. One change was made before the run: the layer4 variants were capped at 400 steps and checked every 10 steps (every 50 after step 100), to fit the time budget. **Result:** at matched hit rates, training still has lower AP (0.47–0.54) and lower dev-class top-1 (47–53%) than the agents (0.57–0.67 and 57–60%). The practitioner rule picks step 0 for layer4 + fc and leaves the fc variants at 41–43% held-out hit. No checkpoint passes. See RESULTS.md.

### Amendment A3, 2026-11-04 ~20:50: DFR-style retraining and a scripted version of the agents' recipe (POST-HOC)

Both arms run on cached penultimate features, in `vision_spurious/a3_arms.py`. No eval data is used in any selection, and both are scored with the original held-out metrics and pass bar.

**1. `dfr_lam*`: DFR-style balanced last-layer retraining** (after Kirichenko et al. 2022).
- **Start:** the pretrained fc, with an L2 penalty toward the original weights and bias. We don't retrain from scratch, because about 40 photos per class is too few to relearn 1,000 classes.
- **Optimizer:** full-batch Adam, lr 1e-4, 300 steps.
- **Loss terms:**
  - CE on all pool-train photos plus the dev photos, to keep the other classes stable.
  - Weighted CE on the dev classes' pool-train photos. A photo gets weight 2 when its "cue score" is above its class median, else 1. The cue score is the summed contribution of that class's top-50 cue channels, the session-2 recipe's definition, computed from the dev cue images and pool photos.
  - The mean −log(1 − p_k) over the dev cue images.

  So the class's real photos, with those that contain the cue oversampled, and its cue-only images carry equal total weight.
- **Grid:** λ ∈ {0, 0.01, 0.1, 1}.
- **Selection:** A2's practitioner rule (3-fold cue CV plus pool-val damage limits), applied over all (λ, step) pairs.
- **Matched points:** for each agent target, the λ whose first matched step has the best pool-val dev-class top-1.

**2. `script_*`: the agents' shared recipe, closed form.** For each dev class k:
- **Cue channels:** w_kj > 0 and mean on k's cue images ≥ r × the clean mean. The top m, ranked by w_kj(μ_s − μ_c), are scaled by (1 − a).
- **Object channels:** defined symmetrically. The top m are scaled by (1 + b).
- **Fixed variants:**
  - **s2 settings:** r = 1.5, m = 50, a = 1, b = 0.5.
  - **s1 settings:** r = 2, m = 32, a = 0.75, with b chosen in closed form to restore the lost clean logit, capped at 1. Session 1 used r = 2 per its notes; the coordinator's description said 1.5.
- **Clean means:** either all of the ImageNet-pool photos of k (`_pool`), or the 3 dev photos (`_dev3`, the agents' own data). This separates "recipe" from "more data".

No tuning.


**Result (run 2026-11-04 ~20:55).**
- **Scripted recipe:** `script_s2_dev3` reproduces agent s2 exactly, and `script_s1_dev3` reproduces agent s1 to within 0.1 point. With pool means, both variants fail the bar: s2 fails conditions 1 and 2, s1 fails condition 1.
- **DFR:** the practitioner rule picked λ = 0.1 at step 300, the step cap. That model passes conditions 1 and 2: hit 21.5%, AP 0.583, −0.29 overall, 62.9% on the dev classes. It is the only arm that passes. Its matched-s3 point (λ = 0.1, step 170) also passes.

Details are in RESULTS.md.
