# Plan (2026-11-04): does an agent's mid-training edit give a ViT locality, and does the gain last?

> **Outcome (2026-11-04 ~19:40): no.** The agent arm ended level with no edit: mean −0.10 points at step 4,000 (per seed −1.45, +0.12, +1.03), with no head start (it reached 75% 0–500 steps later). Conditions 1 and 3 failed. The random edit gave −0.20. Mimetic at step 200 gave −1.52 and was worse in every seed, so the barrier test failed too. The written locality partly survived training (early-layer attention distance 3.1–3.4 against 3.6–3.7), but it bought no accuracy. Full tables are in [RESULTS.md](RESULTS.md) and the summary is in [FINDINGS.md](FINDINGS.md). Everything below is the plan as written.

## Background

- **Induction-plateau pretraining run** ([FINDINGS.md](FINDINGS.md), "agent edits during small-LM pretraining"): an Opus 5.5 edit at step 1,000 wrote a previous-token and induction circuit and led by up to 0.33 nats (about 650 steps), but AdamW formed its own circuit and the final gap was noise. A plateau makes the window longer, not permanent.
- **Open question** ([NEXT_PLAN.md](NEXT_PLAN.md), "Still open"): *barriers rather than plateaus*, i.e. changes the loss would reward but the optimizer can't reach within the budget.
- **Why vision:** a ViT trained from scratch on a small dataset like CIFAR-10 is known to learn local attention slowly and incompletely, and inductive biases for locality (convolutional stems, locality-biased attention, mimetic initialization) raise its final accuracy by several points. That gap at the *end* of training is a candidate barrier: the data-limited optimizer never builds the structure, so a written-in structure might keep paying until the end.

## Hypothesis

A ViT trained from scratch on CIFAR-10 learns local attention slowly and incompletely. An agent edit at about 5% of training that writes locality or positional structure into attention gives a head start, and, unlike the LM run, the gain may **last to the final test accuracy**, because this is a barrier rather than a plateau.

## Setup (fixed after one quick check; see "Recipe check")

- **Model** (`vision/model.py`): ViT, patch 4 on 32×32 → 8×8 = 64 tokens, no class token (mean pool), width 192, **6 blocks**, 3 heads of size 64, MLP 768 (GELU), pre-LayerNorm, learned absolute `pos_embed`, fused `qkv` with bias. 2.69M parameters.
- **Data** (`vision/data.py`): CIFAR-10 (50k train / 10k test), held on the GPU as uint8; random crop from the 4-pixel zero-padded image + horizontal flip, done on the GPU. Batch order and augmentation depend only on (data seed, step), so resumed arms see identical batches.
- **Recipe** (`vision/train.py`): AdamW (0.9, 0.999), weight decay 0.05 on matrices, grad clip 1.0, bf16, `torch.compile`; lr 1e-3, 200-step warmup, cosine to 1e-5; batch 512 (97 steps per epoch); **4,000 steps ≈ 41 epochs**. No label smoothing, mixup, dropout or stochastic depth.
- **Probes every 100 steps:** full test set accuracy and loss; 5,000 fixed training images; per-head mean attention distance, 3×3-neighbourhood mass and self mass on 500 test images.
- **Edit step:** **step 200 of 4,000 (5%)**, the end of warmup, about 2 epochs in.
- **Seeds:** 0, 1, 2 (init seed = data seed). Base runs train to step 200 and save the checkpoint with optimizer state; every arm, including the control, resumes from that checkpoint.

## Arms (all resumed from the same step-200 checkpoint, same optimizer state, same batches)

| Arm | What is written at step 200 | Seeds |
|---|---|---|
| `none` | nothing (copy of the checkpoint) | 0, 1, 2 |
| `agent` | one blind Opus 5.5 session per seed (`runs/vision_vit/s{seed}_step200/PROMPT.md`) | 0, 1, 2 |

Control checkpoints and all `arm_*` runs live in `runs/vision_vit/arms/s{seed}_step200/`, outside the agent's folder.
| `random0` | random edit matched to that seed's agent edit per tensor: same tensors, same number of changed entries, same L2 norm of change, random positions and directions (`agent_session.py random`, same logic as `pretrain/agent_session.py`) | 0, 1, 2 |
| `mimetic` | **scripted reference fix**: mimetic attention (Trockman & Kolter 2023) added to every layer, see below | 0, 1, 2 |

Dropped for time (user constraint, 2026-11-04): mimetic at step 0, and extra seeds for the cheap arms.

### The scripted reference fix (`vision/mimetic.py`), exactly

Per layer and per head h (x is the LayerNormed token, d = 192, dh = 64, Z has i.i.d. N(0, 1/d) entries, fresh per head):
- **QK:** M = Wq_hᵀ Wk_h (the head's d×d score matrix) + 0.7·Z₁ + 0.7·I. SVD M = U S Vᵀ; keep the top 64: Wq_h = S^½ Uᵀ, Wk_h = S^½ Vᵀ.
- **VO:** N = Wo_h Wv_h (the head's d×d output map) + 0.4·Z₂ − 0.4·I. SVD; keep the top 64: Wo_h = U S^½, Wv_h = S^½ Vᵀ.
- Biases, LayerNorms, MLPs, `pos_embed` and the patch embedding are untouched. Every entry of every `qkv.weight` and `proj.weight` changes (884,736 entries, 33% of the model).
- α = β = 0.7 (QK) and 0.4 (VO) are the paper's ViT values as we recall them. Mid-training we *add* the mimetic structure to what each head has learned and refactor at the same rank; at step 0 this reduces to the paper's init. Known deviations: per-head rank-64 truncation of the identity for VO as well as QK, no position-embedding change, additive use mid-training.

### Agent rules (`vision/agent_tools.py`, mirroring `runs/pretrain_agent/code_d512_s0_step1000/PROMPT.md`)

- Sees the step-200 weights, the **training set only** (a train-only data file; the test set is never reachable through the tools), the recipe and the training loss/accuracy history so far.
- Tools: forward passes with logits, per-image loss, attention maps, residual stream, per-head and per-MLP outputs; `metrics` (training-image loss/accuracy and per-head attention stats). 1,000 counted forward calls.
- Writes values through `save_edit` with scope and budget checks: **budget 900,000 changed entries (33% of 2.69M)**, chosen to allow rewriting every attention matrix in every block (884,736), as the scripted fix does. `patch_embed.*` and `head.*` are out of scope.
- No optimizer, gradients or fitting; a handful of hand-chosen variants is fine. **About 30 minutes per session**, stated in the prompt.
- Told it will be scored on final test accuracy (main), mean accuracy over the span and steps to fixed levels, against no edit and a matched random edit, plus "a scripted reference intervention from the literature" (not named). Not told about locality.

## Metrics

- **Primary:** final test accuracy at step 4,000. Also **last-5** (mean of the last 5 probes, steps 3,600–4,000), less noisy.
- **Head start:** first step at which test accuracy reaches 60/70/75/80% (levels adjusted to the recipe check; see below), and mean test accuracy over steps 200–4,000.
- **Locality diagnostics:** mean attention distance per head and 3×3 mass over training, to see whether the written structure survives.

## Pass bar (fixed 2026-11-04 ~19:10, after the recipe check and before any arm result was looked at)

Let Δ_X = (arm X − none) in test accuracy at step 4,000, per seed, in points. The recipe check (below) puts the
whole locality headroom at about **+0.9 points** (a hard-wired locality bias vs plain) and the seed-to-seed spread of
plain runs at about 0.2 points, so the bar is set at roughly half the headroom.

1. **Lasting gain (the hypothesis):** mean Δ_agent ≥ **+0.5 points** and Δ_agent > 0 in **3 of 3** seeds, on the
   final accuracy **and** on last-5.
2. **Control:** the random arm does not meet condition 1.
3. **Head start:** the agent arm reaches **75%** test accuracy at least **30% sooner** than none, counted in steps
   after the edit (step 200), in at least 2 of 3 seeds. (Plain runs reach 75% at about step 2,000, so the agent
   needs about step 1,460 or earlier.)
4. **Beats the scripted fix:** mean (agent − mimetic) ≥ **+0.3** on final accuracy and positive in at least 2 of 3
   seeds. Reported separately.
- **"The agent helps"** means 1 and 2 hold. "Head start only" means 3 holds without 1.
- **Barrier test independent of the agent:** if `mimetic` meets condition 1 itself, a mid-training locality edit
  gives a lasting gain here (barrier supported). If neither lasts, the LM run's plateau story carries over to vision.
- **Diagnostics reported regardless:** per-layer mean attention distance and 3×3 mass at steps 200, 1,000 and
  4,000 for every arm, to show whether written structure survives training.

## Predictions (written before running)

- The plain ViT's attention stays broad (mean distance near the uniform 4.1 in most layers), confirming that it doesn't learn locality within 41 epochs.
- The agent will notice that attention is non-local and position structure is weak, and write locality, most likely through `pos_embed` similarity plus Q/K rows that read position (a relative-position-like bias), possibly also identity-like QK. Immediate effect on training loss: small or slightly negative.
- Head start: agent and mimetic reach 70–75% a few hundred steps sooner than none; condition 3 (30% sooner) is a coin flip, about 50%.
- Lasting gain: **I expect the gain to fade or be too small to see**: mean Δ_agent between −0.3 and +0.5 points, so condition 1 most likely fails (about 70%). Mimetic about the same as none at the end (within ±0.4). Even a perfectly built locality structure has only about +0.9 points of headroom in this recipe. The reason for not expecting a large lasting gain: with weight decay 0.05 on all attention matrices and no positional gradient pressure, written structure decays unless the loss supports it, and the from-scratch ViT already reaches high training accuracy, so the remaining gain is a generalization effect the training loss can't see.
- Random edit: within ±0.5 of none at the end.

## Recipe check (2026-11-04, ~19:00; runs in `runs/vision_vit/check/`, pilot seeds 100/101, not experiment seeds)

One quick check, three runs at once, full recipe (depth 6, 4,000 steps):

| Run | Final test acc | Last-5 | Step reaching 60 / 70 / 75 / 80% | Mean attention distance per layer at the end (uniform = 4.14) | 3×3 mass per layer (uniform ≈ 0.12) |
|---|---|---|---|---|---|
| plain, seed 100 | 81.69 | 81.64 | 700 / 1,400 / 2,000 / 3,100 | 3.74 3.39 3.49 3.87 4.12 4.13 | 0.17 0.20 0.21 0.13 0.09 0.08 |
| plain, seed 101 | 81.51 | 81.43 | 700 / 1,400 / 2,000 / 2,800 | 3.73 3.60 3.42 3.96 3.90 3.99 | 0.18 0.16 0.23 0.12 0.13 0.09 |
| fixed Gaussian locality bias (σ = 1, 2, 4 patches for heads 0–2, every layer), seed 100 | **82.53** | 82.44 | 700 / 1,200 / 1,900 / 2,500 | 2.24 2.35 2.48 2.51 2.70 3.07 | 0.45 0.44 0.35 0.35 0.32 0.23 |

- **The plain ViT barely learns locality:** its attention stays near uniform distance in layers 3–5 and only mildly local in layers 0–2.
- **But the locality gap is small in this short recipe: about +0.9 points** at the end (one seed of each), and about 500 steps to 80%. A gap of several points is what the literature reports for longer, more data-limited training; 41 epochs is not enough overfitting for locality to matter much. As instructed, noted and kept. This caps what any locality edit can win here, and the pass bar is scaled to it.
- An earlier, aborted pilot at depth 9 / 20,000 steps (about 15 min per run) was stopped when the time budget changed; it showed the same picture at step 3,000 (attention distances 3.5–4.2).

## Amendments

- **2026-11-04 ~18:55, before any arm ran (user constraint: about 2.5 h wall-clock for the whole experiment).** The first draft used depth 9 and 20,000 steps (about 15 min per run, edit at step 1,000), 5 seeds for cheap arms and mimetic at step 0. Changed to depth 6, 4,000 steps (about 3–4 min per run solo), edit at step 200, 3 seeds for every arm, no mimetic-at-step-0 arm, runs executed concurrently, 30-minute agent sessions. The pass bar above was written after this change and after the recipe check. The seed-0 `none` and `mimetic` arms had been launched at 19:06, a few minutes before the bar was written down; none of their output was read before it was fixed.
- **Data source.** CIFAR-10 came from the Hugging Face mirror `uoft-cs/cifar10` (parquet with PNG bytes, `vision/data_hf.py`) because the official server delivered about 130 KB/s through torchvision. Same 50,000/10,000 split, 5,000/1,000 images per class.
