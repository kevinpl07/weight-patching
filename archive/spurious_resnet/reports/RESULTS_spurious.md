<!-- Extracted from RESULTS.md (sections matching ['spurious features']); full file in archive/reports/RESULTS.md -->

# Results: training-free repair of spurious features in ResNet-50 (Experiment 2, 2026-11-04)

Plan, pass bar and predictions (fixed before any arm ran): [PLAN_2026-11-04_spurious_repair.md](PLAN_2026-11-04_spurious_repair.md). Code: `vision_spurious/`. Runs: `runs/vision_spurious/`.

## Setup

- **Model:** torchvision ResNet-50, `IMAGENET1K_V1`, frozen. The edit is final.
- **Failure:** Spurious ImageNet (Neuhaus et al., ICCV 2023): 100 classes × 75 photos that show only a class's spurious cue.
- **Clean set:** ImageNetV2 matched-frequency.

**Splits, seed 0.** The agents and every fitted or scripted baseline see only dev:
- Spurious images of 50 "dev classes", 25 per class (1,250).
- 3 ImageNetV2 photos per class (3,000).

Eval is held out:
- 50 new spurious images per dev class (2,500).
- 7 ImageNetV2 photos per class (7,000).
- All 75 images of each of the other 50 "unseen" classes (3,750).

**Metrics:**
- **Hit:** the share of spurious images predicted top-1 as their class.
- **AP:** the benchmark's spurious score. It is the average precision of p_k, ranking clean photos of k above spurious photos of k. Here it has only 7 positives per class, so it isn't comparable with the paper's table.
- **Top-1:** on the 7,000 ImageNetV2 eval photos.

**Agents:** three blind Opus 5.5 sessions. Each could edit up to 200,000 entries in any float tensor, had 300,000 forward images and about 30 minutes, and could use no gradients or fitting.

## Main table (held-out eval)

| Arm | Entries changed | Dev hit | Dev AP | Dev ROC AUC | Top-1 | Δ top-1 (paired SE) | Top-1, dev classes (350) | Unseen hit | Unseen AP |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 50.2% | 0.476 | 0.621 | 62.91% | – | 62.3% | 45.1% | 0.554 |
| **agent s1** | 3,169 | 21.8% | 0.570 | 0.717 | 62.79% | −0.13 (0.07) | 57.1% | 47.1% | 0.548 |
| **agent s2** | 5,000 | **15.7%** | **0.670** | **0.801** | 62.34% | −0.57 (0.13) | 59.7% | 46.9% | 0.538 |
| **agent s3** | 29,912 | 17.2% | 0.594 | 0.742 | 62.77% | −0.14 (0.09) | 58.9% | 47.5% | 0.545 |
| random, same entries s1 / s2 / s3 | same as agent | 48.4 / 48.3 / 48.5% | 0.477 / 0.476 / 0.473 | 0.62–0.63 | 62.80–62.94% | −0.11 to +0.03 | 62.6 / 60.6 / 62.9% | 43.6–45.4% | 0.553–0.557 |
| random, same count s1 / s2 / s3 | same as agent | 49.9 / 49.7 / 49.6% | 0.472–0.475 | 0.62 | 62.79–63.03% | −0.11 to +0.13 | 61.7 / 61.7 / 62.9% | 43.3–44.6% | 0.553–0.554 |
| proj (scripted fc projection) | 102,450 | 0.2% | 0.772 | 0.889 | 61.49% | −1.43 (0.18) | **24.6%** | 48.1% | 0.553 |
| spufix_like (SpuFix clamp, wrapper) | 0 | 0.1% | 0.644 | 0.768 | 60.99% | −1.93 (0.20) | 13.7% | 48.1% | 0.553 |
| llr (last-layer retraining), trained to ≤2% dev hit | 2,049,000 | 5.4% | 0.498 | 0.648 | 61.04% | −1.89 (0.38) | 38.6% | 43.7% | 0.554 |
| ft_l4 (layer4 + fc fine-tune), trained to ≤2% dev hit | 17,013,698 | 4.6% | 0.619 | 0.759 | 61.94% | −0.99 (0.34) | 50.3% | **35.9%** | 0.570 |

The paired SE is over the 7,000 clean eval photos. The dev-class top-1 is measured on 350 photos, so its unpaired SE is about 2.6 points.

**Note.** The llr and ft_l4 rows sit at an extreme operating point, because they were trained until the dev hit rate was at most 2%. The fair comparison at the agents' own operating points is in "Amendment A1" below.

**Per class** (50 dev classes; AP counts as up or down when it moves by more than 0.01; top-1 is measured on each class's 7 clean photos):

| Arm | AP up | AP down | Class-k top-1 down | Class-k top-1 up |
|---|---|---|---|---|
| s1 | 45 | 0 | 15 | 0 |
| s2 | 48 | 0 | 14 | 13 |
| s3 | 48 | 1 | 13 | 6 |
| proj | 48 | 2 | 45 | 0 |
| ft_l4 | 45 | 3 | 31 | 3 |

## Pass bar

The thresholds come from the base model:
1. **Fixes the failure:** dev hit ≤ 25.1% and dev AP ≥ 0.576.
2. **No collateral damage:** top-1 ≥ 62.41% and dev-class top-1 ≥ 59.29%.
3. **Not just any perturbation:** the matched same-entries random edit fails condition 1.

| Session | 1: fixes the failure | 2: no collateral damage | 3: random control fails 1 | Pass |
|---|---|---|---|---|
| s1 | no: AP 0.570 is 0.006 short | no: dev-class 57.1%, 2.2 points short | yes | **no** |
| s2 | yes | no: top-1 62.34%, **0.07 points (5 photos) short**; dev-class OK | yes | **no** |
| s3 | yes | no: dev-class 58.9%, **0.4 points (about 1–2 photos of 350) short**; overall OK | yes | **no** |

**0 of 3 sessions pass, so the experiment fails its pass bar.** Two sessions miss by margins far below the metrics' noise. No baseline meets condition 2 either; they miss by 1–2 points overall and 12–49 points on the dev classes.

## Agents' own predictions against the outcome

| Session | Edit | Predicted | Held-out outcome |
|---|---|---|---|
| s1 | `fc.weight` only, 3,169 entries (‖Δ‖ = 2.04). Per dev class: the top 32 cue channels ×0.25, plus up to 32 object channels boosted to restore the clean-k logit. | hit 0.20–0.26, AP +0.05–0.08, top-1 ±0.2, dev-class −3 to −6 | hit 0.218 ✓, AP +0.094 (better), top-1 −0.13 ✓, dev-class −5.1 ✓ |
| s2 | `fc.weight` only, 5,000 entries (‖Δ‖ = 4.27). Per dev class: the top 50 cue channels zeroed and the top 50 object channels ×1.5. | hit 0.17–0.25, AP 0.68–0.72, top-1 −0.3 to +0.2, dev-class −1 to −3 | hit 0.157 (better), AP 0.670 (just below), **top-1 −0.57 (worse)**, dev-class −2.6 ✓ |
| s3 | `fc.weight` 29,862 + `fc.bias` 50. A graded cut of cue channels (by how much more they fire on cue-only images) and a graded boost of object channels; half of the clean-k logit change is restored through the bias. | hit 0.18–0.25, AP 0.58–0.62, top-1 −0.3 to +0.2, dev-class −3 to −6 | hit 0.172 (slightly better), AP 0.594 ✓, top-1 −0.14 ✓, dev-class −3.4 ✓ |

**10 of the 12 predicted intervals contain the outcome or err on the good side.** The two misses:
- s2's top-1: −0.57 against a predicted −0.3 to +0.2. Its ×1.5 object-channel boost lets dev classes steal predictions from other classes: top-1 on the 900 classes outside the benchmark fell from 62.68% to 62.21%.
- s2's AP: 0.670, just below its predicted 0.68–0.72.

Every session estimated dev-class damage with cross-validation on its 3 clean photos per class, and those estimates held. The in-sample dev-class numbers (s2: 72.7%, s3: 66.0%) were optimistic, as the agents themselves wrote.

## Findings

1. **All three sessions found the same mechanism and built the same kind of edit, independently.**
   - They cached penultimate features and defined "cue channels" for class k: channels with w_kj > 0 that fire much more on k's spurious images than on k's clean photos.
   - They checked that these channels are context detectors. The clean images that fire them most are sharks and coral for snorkel; envelopes and menus for quill and fountain pen; daisies for hummingbird and bee; skis for snowplow.
   - They then shrank those weights in fc row k and boosted the "object" channels to compensate.
   - All three edits are fc-only and change 0.01–0.12% of the model.
2. **The agents' edits dominate every baseline on the trade-off between the fix and the damage it causes.**
   - s2 reaches a higher dev AP than the layer4 fine-tune (0.670 against 0.619), with about half its top-1 loss (−0.57 against −0.99), about a fifth of its dev-class loss (−2.6 against −12.0 points), and 3,400× fewer changed entries.
   - The scripted projection gets the best AP (0.772) only by wiping out the classes themselves: dev-class top-1 falls to 24.6%. AP alone is gameable by suppressing class k globally, which is why the plan pairs it with dev-class top-1.
3. **Training on the same small dev set is worse than editing.** This holds at the agents' own operating points too: see Amendment A1 below.
   - Last-layer retraining hit its dev stopping rule (dev hit ≤ 2%) after 62 Adam steps, but held-out AP barely moved (0.476 → 0.498). It mostly learned to suppress class k (dev-class top-1 −24 points) and fit the 3 clean photos per class (dev clean top-1 64% → 83%).
   - The layer4 fine-tune generalizes better, but costs 1 point overall and 12 points on the dev classes.
4. **Random edits do nothing.** Random edits of the same size, at the same entries or at random positions, leave hit and AP within ±2 points and ±0.004.
5. **Unseen classes: no transfer, and slight harm.**
   - Each agent edit raises the unseen-class hit rate by 1.8–2.4 points. Random edits move it by −1.8 to +0.3, so this is at the edge of noise.
   - A likely reason: suppressing dev classes removes competitors for the unseen classes' spurious images. proj shows the same effect (+3.0).
   - Only ft_l4, which changes layer4 features, reduces the unseen hit rate (45.1% → 35.9%).
6. **The pass bar was strict relative to the noise.** The plan used 0.5 points for overall top-1, where the paired SE is 0.1–0.13. It used 3 points for dev-class top-1, measured on 350 photos with SE about 2.6. s2 and s3 miss by 5 photos and about 1–2 photos.

### The plan's predictions against the outcome

| Arm | Predicted | Outcome |
|---|---|---|
| Base | top-1 about 63%, AP 0.4–0.55 | correct (62.9%, 0.476) |
| Base hit rate | 30–45% | higher: 50% |
| proj | dev-class cost 1–3 points | wrong: −38 points |
| llr | the largest fix, with top-1 unchanged | wrong: no AP gain and −1.9 top-1 |
| ft_l4 | −1 to −3 points of top-1 | correct (−1.0) |
| Agents | 2 of 3 pass, AP comparable to proj | 0 of 3 pass by thin margins, lower AP than proj but far less damage |

## Amendment A1 (post-hoc): training stopped at the agents' operating point

**Why.** The original llr and ft_l4 arms were trained until the in-sample dev hit rate was at most 2%. The agents instead compared doses and stopped at an in-sample dev hit rate of 0.234 (s1), 0.165 (s2) and 0.173 (s3). The question here: **would plain training on the same images, stopped at the same point, have done the agents' job?**

**Setup** (`vision_spurious/train_curves.py`; the amendment is in the plan). Five variants, each started from the pretrained weights and trained on exactly the agents' dev data, with the loss CE(clean) − log(1 − p_k)(spurious):

| Variant | Trains | Optimizer | Extra |
|---|---|---|---|
| `llr` | fc | full-batch Adam, lr 1e-4 | – |
| `llr_rows` | only the 50 dev-class rows of fc, the agents' scope | full-batch Adam, lr 1e-4 | – |
| `llr_gentle` | fc | Adam, lr 2e-5 | + KL(original ‖ new) on all 3,000 clean dev images |
| `ft` | layer4 + fc | Adam, lr 1e-5, batch 64 | – |
| `ft_gentle` | layer4 + fc | Adam, lr 1e-5, batch 64 | + KL on each batch's clean images |

**Checkpoint selection used dev only:**
- **match sN:** the first step whose in-sample dev hit rate falls to agent sN's.
- **3-fold CV:** the lowest CV hit rate among steps that stay within the pass bar's damage limits on held-out dev folds.

**"Train s"** below is pure GPU training time, excluding feature caching and evaluation.

| Arm | Step | Train s | Dev hit (in-sample) | Held-out hit | Held-out AP | Δ top-1 | Top-1, dev classes | Unseen hit | Conditions 1 / 2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | – | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| **agent s1** | – | – | 0.234 | 21.8% | **0.570** | −0.13 | **57.1%** | 47.1% | no / no |
| llr @ match s1 | 11 | 0.07 | 0.230 | 23.5% | 0.476 | −0.60 | 50.0% | 45.5% | no / no |
| llr_rows @ match s1 | 21 | 0.11 | 0.231 | 23.4% | 0.471 | −0.29 | 49.1% | 47.7% | no / no |
| llr_gentle @ match s1 | 65 | 0.39 | 0.231 | 23.9% | 0.481 | −0.04 | 51.1% | 46.3% | no / no |
| ft @ match s1 | 28 | 0.87 | 0.223 | 23.4% | 0.544 | −0.90 | 52.6% | 41.1% | no / no |
| ft_gentle @ match s1 | 30 | 0.97 | 0.230 | 24.5% | 0.546 | −0.23 | 54.3% | 44.0% | no / no |
| **agent s2** | – | – | 0.165 | **15.7%** | **0.670** | −0.57 | **59.7%** | 46.9% | yes / no |
| llr @ match s2 | 15 | 0.09 | 0.163 | 17.3% | 0.478 | −0.81 | 48.0% | 44.9% | no / no |
| llr_rows @ match s2 | 29 | 0.15 | 0.162 | 17.2% | 0.476 | −0.37 | 46.3% | 48.0% | no / no |
| llr_gentle @ match s2 | 94 | 0.56 | 0.165 | 18.3% | 0.485 | −0.31 | 48.6% | 46.4% | no / no |
| ft @ match s2 | 42 | 1.30 | 0.165 | 20.3% | 0.555 | −0.74 | 53.1% | 41.1% | no / no |
| ft_gentle @ match s2 | 50 | 1.62 | 0.164 | 20.4% | 0.562 | −0.41 | 55.1% | 43.2% | no / no |
| **agent s3** | – | – | 0.173 | **17.2%** | **0.594** | −0.14 | **58.9%** | 47.5% | yes / no |
| llr @ match s3 | 15 | 0.09 | 0.163 | 17.3% | 0.478 | −0.81 | 48.0% | 44.9% | no / no |
| llr_rows @ match s3 | 28 | 0.14 | 0.168 | 17.6% | 0.472 | −0.34 | 46.9% | 48.0% | no / no |
| llr_gentle @ match s3 | 90 | 0.54 | 0.171 | 18.7% | 0.485 | −0.30 | 48.9% | 46.4% | no / no |
| ft @ match s3 | 40 | 1.24 | 0.171 | 20.4% | 0.553 | −0.70 | 53.1% | 41.2% | no / no |
| ft_gentle @ match s3 | 48 | 1.55 | 0.168 | 20.8% | 0.561 | −0.36 | 55.1% | 43.0% | no / no |
| llr @ 3-fold CV | 2 | 0.03 | 0.469 | 44.8% | 0.475 | +0.24 | 60.3% | 45.5% | no / yes |
| llr_rows @ 3-fold CV | 4 | 0.02 | 0.463 | 44.2% | 0.473 | +0.03 | 60.0% | 46.0% | no / yes |
| llr_gentle @ 3-fold CV | 15 | 0.09 | 0.434 | 42.6% | 0.475 | +0.20 | 59.4% | 45.6% | no / yes |
| ft @ 3-fold CV | 6 | 0.19 | 0.415 | 40.7% | 0.490 | −0.43 | 56.9% | 43.7% | no / no |
| ft_gentle @ 3-fold CV | 8 | 0.26 | 0.387 | 38.8% | 0.497 | −0.63 | 56.3% | 44.1% | no / no |

**Oracle check** (diagnostic only; it selects with held-out data, which the rules forbid). Across the whole held-out trajectory of every variant:
- No training checkpoint meets conditions 1 and 2 together.
- The best dev-class top-1 at a held-out hit rate of 25% or less is 55.7% (`ft_gentle`, step 60), against 57.1–59.7% for the agents.
- The highest AP anywhere is 0.609 (`ft`, at 6.8% hit and 51.7% dev-class top-1), against the agents' 0.570–0.670.
- Plot: `runs/vision_spurious/tradeoff.png`. Its x-axis is held-out hit; its y-axes are dev-class top-1 and overall top-1. The agents sit 2–5 points above the best training curve (`ft_gentle`) and about 10 points above the fc curves on dev-class top-1 at equal hit rate.

**Findings of the amendment:**
1. **At equal hit rates, the agents' edits beat plain training on what matters for the reported classes.**
   - **Spurious AP:** 0.57–0.67 for the agents, against 0.47–0.49 for fc training and 0.54–0.56 for layer4 training.
   - **Dev-class top-1:** 57–60% for the agents, against 46–51% for fc training and 53–55% for layer4 training.
   - **Why:** fc training reaches the agents' hit rate mostly by lowering class k everywhere, so AP doesn't move. The agents removed specific context channels and compensated through object channels.
2. **On overall top-1, gentle training is as good as the careful agents.** `llr_gentle` costs −0.04 to −0.31 and `ft_gentle` −0.23 to −0.41, against s1 and s3 at −0.13/−0.14. Both gentle variants beat s2's −0.57. The edits' advantage is concentrated on the reported classes, not on the other 950.
3. **Only layer4 training transfers to unseen classes.** It cuts their hit rate by 1–4 points (41–44% against 45.1%). fc training and the agents' fc edits don't transfer, and slightly raise it.
4. **Selecting a training checkpoint by 3-fold CV on dev barely trains at all** (steps 2–15, held-out hit 39–45%). With 3 clean photos per class, the damage limits bind almost immediately: training can't find a point that both fixes the failure and passes the damage check on dev. The agents did find one, using the same 3 photos, and their CV estimates held on eval.
5. **Training is far cheaper.** The matched points take 11–94 steps and 0.07–1.6 s of GPU training. That is about 2 minutes end to end, including a one-off feature pass over dev, against about 4–5 minutes and many tokens per agent session. But the cheap result is worse.
6. **The pass bar is unchanged.** No training checkpoint passes condition 2 at any operating point that meets condition 1. The agents remain 0 of 3, by thin margins.

## Amendment A2 (post-hoc): training with ImageNet-train data

**Why.** A1 trained on 3 clean photos per class. A2 gives training a realistic amount of labelled data, as every earlier experiment's training had.

**Data.** The user accepted the ImageNet terms and downloaded 12 `ILSVRC/imagenet-1k` train shards (5.6 GB). `vision_spurious/imagenet_pool.py` read them as data only (`python -I`) and kept up to 50 photos per class, preprocessed like eval:
- **Pool size:** 48,092 photos. Per class: minimum 23, 5th percentile 41, median 50. Dev classes: minimum 39, median 50.
- **Pool-val:** the first min(10, n/5) per class are held out (9,445 photos). The rest form pool-train (38,647).

**Training** (`vision_spurious/train_full.py`).
- **Positives:** pool-train plus the 3 ImageNetV2 dev photos per class.
- **Negatives:** the same 25 dev cue-only images per class.
- **Loss:** the A1 form, with each term averaged over its own images.
- **Variants:**
  - `ft_gentle_in` and `ft_in`: layer4 + fc, lr 1e-5, 64 positives + 16 negatives per step, at most 400 steps; the first with KL to the original model.
  - `llr_gentle_in`: fc, full batch, lr 2e-5, + KL.
  - `llr_rows_in`: the 50 dev-class fc rows only, full batch, lr 1e-4.

**Stopping, without eval data:**
- **match sN:** as in A1.
- **practitioner rule:**
  - Three runs, each trained on 2/3 of every class's cue images.
  - Choose the step with the lowest CV cue hit rate, among steps where pool-val top-1 is no more than 0.5 points below base and pool-val dev-class top-1 no more than 3 points below.
  - Score the full run at that step.

| Arm | Step | Train s | Dev hit (in-sample) | Held-out hit | Held-out AP | Δ top-1 | Top-1, dev classes | Unseen hit | Conditions 1 / 2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | – | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| **agent s1** | – | – | 0.234 | 21.8% | **0.570** | −0.13 | **57.1%** | 47.1% | no / no |
| ft_gentle_in @ match s1 | 20 | 2.1 | 0.231 | 25.2% | 0.522 | −1.10 | 53.4% | 40.5% | no / no |
| ft_in @ match s1 | 20 | 2.1 | 0.222 | 23.5% | 0.521 | −1.60 | 51.4% | 39.0% | no / no |
| llr_gentle_in @ match s1 | 60 | 2.4 | 0.214 | 22.4% | 0.477 | −0.07 | 49.1% | 47.1% | no / no |
| llr_rows_in @ match s1 | 21 | 0.6 | 0.225 | 23.1% | 0.468 | −0.26 | 49.1% | 47.7% | no / no |
| **agent s2** | – | – | 0.165 | 15.7% | **0.670** | −0.57 | **59.7%** | 46.9% | yes / no |
| ft_gentle_in @ match s2 | 40 | 4.3 | 0.142 | 16.4% | 0.544 | −0.84 | 53.4% | 39.3% | no / no |
| ft_in @ match s2 | 40 | 4.2 | 0.123 | 13.6% | 0.543 | −1.34 | 50.6% | 36.6% | no / no |
| llr_gentle_in @ match s2 | 80 | 3.1 | 0.164 | 18.0% | 0.479 | +0.01 | 48.6% | 47.3% | no / no |
| llr_rows_in @ match s2 | 28 | 0.8 | 0.162 | 17.3% | 0.468 | −0.31 | 47.1% | 48.0% | no / no |
| **agent s3** | – | – | 0.173 | 17.2% | **0.594** | −0.14 | **58.9%** | 47.5% | yes / no |
| ft_gentle_in @ match s3 | 30 | 3.2 | 0.172 | 19.4% | 0.534 | −0.94 | 51.7% | 39.3% | no / no |
| ft_in @ match s3 | 30 | 3.2 | 0.166 | 17.7% | 0.536 | −1.67 | 50.0% | 37.1% | no / no |
| llr_gentle_in @ match s3 | 80 | 3.1 | 0.164 | 18.0% | 0.479 | +0.01 | 48.6% | 47.3% | no / no |
| llr_rows_in @ match s3 | 27 | 0.7 | 0.170 | 18.3% | 0.467 | −0.29 | 47.7% | 47.9% | no / no |
| ft_gentle_in @ practitioner rule | 0 (no step qualifies) | 0 | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| ft_in @ practitioner rule | 0 (no step qualifies) | 0 | 0.530 | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| llr_gentle_in @ practitioner rule | 17 | 0.7 | 0.416 | 41.0% | 0.471 | +0.11 | 58.6% | 45.8% | no / no |
| llr_rows_in @ practitioner rule | 5 | 0.1 | 0.438 | 43.0% | 0.473 | +0.01 | 59.4% | 46.3% | no / yes |

**Oracle check** (diagnostic only; it selects with held-out data). Across all four held-out trajectories:
- No checkpoint meets conditions 1 and 2 together.
- The best dev-class top-1 at a held-out hit rate of 25% or less is 54.3% (`ft_gentle_in`, step 90).
- The highest AP is 0.608 (`ft_gentle_in`, step 200), at 7.2% hit, 52.9% dev-class top-1 and −0.50 overall.
- Plot: `runs/vision_spurious/tradeoff_imagenet.png`.

**Findings of A2:**
1. **About 14× more clean data doesn't change the picture.** At each agent's hit rate, training gets:
   - **AP:** 0.47–0.54, against the agents' 0.57–0.67.
   - **Dev-class top-1:** 47–53%, against 57–60%.

   fc training still mostly lowers the class everywhere. Layer4 training with ImageNet data also costs 0.8–1.7 points of overall top-1 at the matched points. That is more than with dev data only (A1), and it recovers only partly with further steps.
2. **Gentle fc training with ImageNet data now ties or slightly beats every agent on overall top-1** (+0.01 to −0.07), but still loses about 10 points on the repaired classes.
3. **The practitioner rule declines to train the layer4 models at all.** For layer4 + fc, no step keeps pool-val within the damage limits, so the rule picks step 0. For fc, it stops at steps 5–17, with held-out hit still 41–43%. A practitioner following a standard validation rule would ship roughly the unfixed model.
4. **Layer4 training is the only route that transfers to the unseen classes,** cutting their hit rate from 45.1% to 37–41%. The agents' fc edits don't.
5. **Cost:** 20–80 steps and 1–4 s of GPU training to reach the matched points. A whole variant took 3–8 minutes, including the 3-fold practitioner runs and evaluation, plus about 3 minutes to build the pool and 3 minutes to cache features.

**Caveats:**
- **ResNet-50 was originally trained on these exact ImageNet-train images.** That is what a practitioner would have, so it's the right counterfactual. But pool-val is not new to the model: base pool-val top-1 is 87.6% against 62.9% on ImageNetV2. So pool-val understates the damage that held-out photos show, and the rule's damage limits are measured on a near-training distribution.
- **Steps were capped** at 400 for layer4 + fc and at 1,500–3,000 for fc, and all four variants reached every matched point.

## Amendment A3 (post-hoc): DFR-style retraining and a scripted version of the agents' recipe

Both arms are in `vision_spurious/a3_arms.py` and run on cached penultimate features. No eval data is used in any selection. The full design is in the plan.

**DFR-style balanced last-layer retraining** (`dfr_lam*`):
- **Start:** the pretrained fc, with an L2 penalty λ toward the original weights.
- **Loss:** three terms of equal weight:
  - CE on all pool-train photos and the dev photos
  - CE on the dev classes' pool photos, with cue-containing photos (cue score above the class median) oversampled 2×
  - −log(1 − p_k) on the 25 cue images per class
- **Grid:** λ ∈ {0, 0.01, 0.1, 1}, full-batch Adam lr 1e-4, 300 steps.
- **Selection:** the practitioner rule (3-fold cue CV plus pool-val damage limits) over all (λ, step) pairs picked **λ = 0.1, step 300**. For the matched points, each agent target takes the λ with the best pool-val dev-class top-1 at its first matched step.

**Scripted recipe** (`script_*`): the agents' shared recipe in closed form, with session 2's and session 1's settings. Clean means come either from the ImageNet pool (`_pool`, ~50 photos per class) or from the 3 dev photos (`_dev3`).

| Arm | Entries / step (GPU s) | Held-out hit | AP | Δ top-1 | Top-1, dev classes | Unseen hit | Conditions 1 / 2 |
|---|---|---|---|---|---|---|---|
| none | – | 50.2% | 0.476 | +0.00 | 62.3% | 45.1% | no / yes |
| agent s1 | 3,169 | 21.8% | 0.570 | −0.13 | 57.1% | 47.1% | no / no |
| agent s2 | 5,000 | 15.7% | 0.670 | −0.57 | 59.7% | 46.9% | yes / no |
| agent s3 | 29,912 | 17.2% | 0.594 | −0.14 | 58.9% | 47.5% | yes / no |
| **script_s1_dev3** | 3,169 | 21.8% | 0.570 | −0.13 | 57.1% | 47.1% | no / no |
| **script_s2_dev3** | 5,000 | 15.7% | 0.670 | −0.57 | 59.7% | 46.9% | yes / no |
| script_s1_pool | 2,527 | 34.1% | 0.534 | +0.04 | 61.7% | 46.5% | no / yes |
| script_s2_pool | 4,792 | 29.1% | 0.653 | −0.76 | **70.0%** | 45.0% | no / no |
| dfr λ=0.01 @ match s1 | step 27 (0.6) | 26.4% | 0.540 | −0.49 | 60.3% | 43.5% | no / yes |
| dfr λ=0.01 @ match s2 | step 47 (1.1) | 22.8% | 0.565 | −0.40 | 62.6% | 43.3% | no / yes |
| dfr λ=0.1 @ match s1 | step 48 (1.1) | 26.3% | 0.549 | −0.39 | 60.3% | 45.8% | no / yes |
| dfr λ=0.1 @ match s3 | step 170 (4.0) | 22.0% | 0.577 | −0.24 | 62.6% | 45.4% | **yes / yes** |
| **dfr λ=0.1 @ practitioner rule** | step 300 (7.1) | 21.5% | 0.583 | −0.29 | **62.9%** | 45.2% | **yes / yes** |

**Other DFR settings:**
- **λ = 0:** the matched points have AP 0.536–0.567 and −0.6 to −0.7 overall.
- **λ = 1:** never reaches any agent's dev hit rate.
- **Oracle check** (held-out, diagnostic only): λ = 0.1 meets both pass-bar conditions at every checked step from 175 to 300, and λ = 0.01 at steps 75 and 125. With λ = 0 at step 300 the AP reaches 0.682 and dev-class top-1 68.0%, but overall top-1 drops 1.63 points.

**Findings of A3:**
1. **The agents' edits are exactly a closed-form recipe.** `script_s2_dev3` reproduces session 2's held-out numbers to the last digit. `script_s1_dev3` reproduces session 1's: 0.218 hit against 0.2184; everything else is identical. Session 3's graded variant is the same idea. Once the recipe is written down, no AI judgment is needed to apply it.
2. **The recipe with better clean statistics trades differently.** With pool means instead of 3 photos, fewer channels qualify as cue channels (4,792 entries changed against 5,000). Hit rates are higher (29–34%), but the repaired classes improve beyond the unfixed model: 70.0% top-1 for the s2 settings, against 62.3%. The s1 settings pass condition 2 but not condition 1. Neither passes.
3. **DFR-style balanced retraining, selected on dev only, is the first arm in this experiment to pass the pass bar.** It has 21.5% hit, AP 0.583, −0.29 overall and no loss on the repaired classes (62.9%). The margin is thin: AP clears 0.576 by 0.007. The practitioner rule picked the last step of the 300-step cap, and the held-out curve was still slowly improving.
   - Compared with the agents, DFR leaves more spurious hits than s2 and s3 (21.5% against 15.7–17.2%), and its AP is below s2's 0.670. But it does no damage to the repaired classes (62.9% against 57–60%).
   - The ingredients that matter, compared with A2's `llr_gentle_in`: balancing each class's real photos (cue-containing ones oversampled) against its cue images, and an L2 pull toward the original fc. With λ = 0, DFR fails condition 2 overall.
4. **DFR is cheap:** 7 s of GPU for the chosen run. The whole grid with CV took about 5 minutes. It needs about 40 labelled photos per class, and its oversampling uses the cue channels defined by the agents' recipe.
5. **No arm transfers to unseen classes except layer4 training.** DFR leaves them at 43–46%.

## Rule audit

- No session script uses `backward`, `autograd`, an optimizer, `lstsq` or `requires_grad`.
- No session script references `eval.pt`, `data_vision/derived`, the baseline scripts or other runs.
- **Forward budget used:** s1 8,500, s2 8,500 and s3 21,250 images, out of 300,000.
- **How the edits were evaluated:** all three agents evaluated their fc-only edits offline from cached penultimate features. That is exact for fc edits, and it is computing with the network, not fitting.
- **Variant comparisons:** hand-chosen dose ladders of 5–9 variants per session, with no automated search.

## Files

- **Code:**
  - `vision_spurious/` (`prepare_data.py`, `common.py`, `eval.py`, `agent_tools.py`, `session.py`, `baselines.py`, `summarize.py`)
  - `vision_spurious/prompt_template.md`
- **Eval outputs:**
  - `runs/vision_spurious/eval/<arm>.json`: summary plus per-class numbers
  - `runs/vision_spurious/eval/<arm>_logits.pt`: fp16 logits
  - `runs/vision_spurious/summary.txt`
- **Sessions:** `runs/vision_spurious/session_{1,2,3}/` (`PROMPT.md`, `work/notes.md`, scripts, `submitted_edit.pt`, `random0_{same,count}_edit.pt`)
- **Baseline weights and training logs:** `runs/vision_spurious/arms/`
- **Amendment A1:**
  - code: `vision_spurious/train_curves.py`, `vision_spurious/summarize_curves.py`, `vision_spurious/plot_tradeoff.py`
  - trajectories and selections: `runs/vision_spurious/curves/<variant>.json`
  - selected checkpoints: `runs/vision_spurious/eval/<variant>_{match_s1,match_s2,match_s3,cv}.json`
  - table: `runs/vision_spurious/summary_curves.txt`
  - plot: `runs/vision_spurious/tradeoff.png`
- **Amendment A2:**
  - code: `vision_spurious/imagenet_pool.py`, `vision_spurious/train_full.py`
  - pool: `data_vision/imagenet_pool/` (shards in `data_vision/imagenet_train_shards/data/`)
  - trajectories: `runs/vision_spurious/curves_in/`
  - selected checkpoints: `runs/vision_spurious/eval/<variant>_in_{match_sN,prac}.json`
  - table: `runs/vision_spurious/summary_curves_in.txt`
  - plot: `runs/vision_spurious/tradeoff_imagenet.png`
- **Amendment A3:**
  - code: `vision_spurious/a3_arms.py`
  - DFR trajectories: `runs/vision_spurious/curves_in/dfr_lam*.json`, `dfr_selection.json`
  - evals: `runs/vision_spurious/eval/{script_*,dfr_lam*}.json`
  - table: `runs/vision_spurious/summary_a3.txt`
  - plot: `tradeoff_imagenet.png`, updated with both arms

