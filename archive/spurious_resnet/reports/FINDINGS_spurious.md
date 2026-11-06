<!-- Extracted from FINDINGS.md (sections matching ['spurious features']); full file in archive/reports/FINDINGS.md -->

## Follow-up: training-free repair of spurious features in an ImageNet classifier (2026-11-04)

**Short answer: the edits work, but they don't win, and they're a recipe a script can apply.** This is the setting the conclusion above asked for: a real failure with a visible symptom, and no training after the edit. Three blind Opus 5.5 sessions each repaired most of a pretrained ResNet-50's reliance on spurious cues by changing 3k–30k fc weights:
- The held-out spurious hit rate fell from 50% to 16–22%.
- Overall clean accuracy fell by 0.1–0.6 points.

None of the sessions passed the bar fixed in advance; they missed by a few photos.

**Two later arms settle the comparison:**
- **Scripted recipe:** a 30-line closed-form version of the recipe all three agents used (cut cue channels, boost object channels in each class's fc row) reproduces sessions 1 and 2 (session 2 exactly, session 1 to within 0.1 point of hit rate).
- **DFR-style retraining** (Kirichenko et al. 2022) of the last layer:
  - **Data:** about 40 labelled ImageNet photos per class, with each class's real photos balanced against its cue images, and an L2 pull toward the original weights.
  - **Selection:** chosen on dev data only.
  - **Result:** it is the only arm that passes the bar, narrowly (AP 0.583 against a 0.576 bar). It leaves somewhat more spurious hits than the best edits (21.5% against 15.7–17.2%) but does no damage to the repaired classes. It needs 7 s of GPU.
- **Naive training** (the earlier A1/A2 arms) loses to the edits, even at the same operating point and with the same ImageNet data.

**Naive training, stopped at the agents' operating point** (chosen without eval data):
- **Variants:** nine in total, including gentle KL-regularized ones and ones restricted to the same 50 fc rows.
- **Data:** first the agents' own dev data, then that plus about 39k labelled ImageNet-train photos.
- **Speed:** it reaches the same hit rate in 11–94 steps, with a few seconds of GPU training.
- **Spurious AP:** lower, 0.47–0.56 against the agents' 0.57–0.67.
- **Repaired classes:** more damage, 46–55% dev-class top-1 against 57–60%.
- **Overall accuracy:** only gentle fc training matches the agents here.

**A realistic stopping rule** (pool-val damage limits plus cross-validated cue hit) stops fc training with almost nothing fixed. It doesn't train the layer4 models at all.

By the pass bar fixed in advance, none of the 3 sessions passes:
- s2 missed the overall-accuracy limit by 5 of 7,000 photos.
- s3 missed the dev-class limit by 1–2 of 350 photos.
- s1 fell short on both the fix and the dev-class limit.

No naive-training checkpoint passes either, even one picked with held-out data; only DFR does. Plan, with post-hoc amendments A1 (fair stopping), A2 (ImageNet-train data) and A3 (DFR and the scripted recipe): [PLAN_2026-11-04_spurious_repair.md](PLAN_2026-11-04_spurious_repair.md); full tables: `RESULTS.md`.

### Setup
- **Model:** torchvision ResNet-50 (`IMAGENET1K_V1`), frozen. The edit is final.
- **Failure:** the Spurious ImageNet benchmark (Neuhaus et al., ICCV 2023). Its photos show only a class's cue, such as flowers for "bee" or underwater scenes for "snorkel", and are still classified as that class.
- **Data:**
  - **Dev,** seen by the agents and all baselines: 25 cue-only photos for each of 50 classes, plus 3 ImageNetV2 photos per class.
  - **Eval,** held out: 50 new cue-only photos per dev class, 7 ImageNetV2 photos per class, and 50 more benchmark classes the agents never saw.
- **Agent:** up to 200k entries (0.78%) of any tensor, forward passes and activations, no gradients or fitting, about 30 minutes.
- **Pass bar:** the hit rate halves and AP rises by 0.10. Overall top-1 drops by no more than 0.5 points and dev-class top-1 by no more than 3. The matched random edit fails the first condition.

### Results (held-out)

| Arm | Entries changed | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit |
|---|---|---|---|---|---|---|
| No fix | 0 | 50.2% | 0.476 | 62.91% | 62.3% | 45.1% |
| **Opus s1 / s2 / s3** | 3,169 / 5,000 / 29,912 | 21.8 / **15.7** / 17.2% | 0.570 / **0.670** / 0.594 | −0.13 / −0.57 / −0.14 | 57.1 / 59.7 / 58.9% | 47.1 / 46.9 / 47.5% |
| Random edits, same entries | same | 48.3–48.5% | 0.473–0.477 | −0.11 to +0.03 | 60.6–62.9% | 43.6–45.4% |
| Scripted projection (fc) | 102,450 | 0.2% | 0.772 | −1.43 | **24.6%** | 48.1% |
| SpuFix-style clamp | wrapper | 0.1% | 0.644 | −1.93 | 13.7% | 48.1% |
| Last-layer retraining, trained to ≤2% dev hit | 2.05M | 5.4% | 0.498 | −1.89 | 38.6% | 43.7% |
| Fine-tune layer4 + fc, trained to ≤2% dev hit | 17.0M | 4.6% | 0.619 | −0.99 | 50.3% | **35.9%** |

The fair comparison stops training where each agent stopped: the first step whose in-sample dev hit rate falls to that agent's (0.234 / 0.165 / 0.173). Ranges below are over the three matched points:

| Training on the agents' dev data | Steps (GPU s) | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit |
|---|---|---|---|---|---|---|
| **Opus s1 / s2 / s3** (for reference) | about 4–5 min sessions | 21.8 / 15.7 / 17.2% | **0.570 / 0.670 / 0.594** | −0.13 / −0.57 / −0.14 | **57.1 / 59.7 / 58.9%** | 47.1 / 46.9 / 47.5% |
| fc | 11–15 (0.1) | 17.3–23.5% | 0.476–0.478 | −0.60 to −0.81 | 48.0–50.0% | 44.9–45.5% |
| fc, 50 dev rows only | 21–29 (0.15) | 17.2–23.4% | 0.471–0.476 | −0.29 to −0.37 | 46.3–49.1% | 47.7–48.0% |
| fc + KL to the original, lr 2e-5 | 65–94 (0.6) | 18.3–23.9% | 0.481–0.485 | −0.04 to −0.31 | 48.6–51.1% | 46.3–46.4% |
| layer4 + fc | 28–42 (1.3) | 20.3–23.4% | 0.544–0.555 | −0.70 to −0.90 | 52.6–53.1% | 41.1–41.2% |
| layer4 + fc + KL | 30–50 (1.6) | 20.4–24.5% | 0.546–0.562 | −0.23 to −0.41 | 54.3–55.1% | 43.0–44.0% |

**With realistic labelled data (amendment A2).** Positives are about 39k ImageNet-train photos (up to 40 per class) plus the dev photos; negatives are the same 25 cue images per class:

| Training with ImageNet-train data | Steps (GPU s) | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit |
|---|---|---|---|---|---|---|
| fc, 50 dev rows only, matched | 21–28 (0.7) | 17.3–23.1% | 0.467–0.468 | −0.26 to −0.31 | 47.1–49.1% | 47.7–48.0% |
| fc + KL, matched | 60–80 (3) | 18.0–22.4% | 0.477–0.479 | +0.01 to −0.07 | 48.6–49.1% | 47.1–47.3% |
| layer4 + fc, matched | 20–40 (4) | 13.6–23.5% | 0.521–0.543 | −1.34 to −1.67 | 50.0–51.4% | 36.6–39.0% |
| layer4 + fc + KL, matched | 20–40 (4) | 16.4–25.2% | 0.522–0.544 | −0.84 to −1.10 | 51.7–53.4% | 39.3–40.5% |
| fc variants, practitioner rule | 5–17 | 41.0–43.0% | 0.471–0.473 | +0.01 to +0.11 | 58.6–59.4% | 45.8–46.3% |
| layer4 variants, practitioner rule | 0 (no step qualifies) | 50.2% | 0.476 | 0 | 62.3% | 45.1% |

`runs/vision_spurious/tradeoff.png` and `tradeoff_imagenet.png` plot every training trajectory on held-out data against the edits. They are diagnostic only. At equal hit rate, the agents sit 2–5 points above the best naive-training curve on dev-class top-1, and about 10 above the fc curves. No naive-training checkpoint on any curve meets the pass bar.

**DFR and the scripted recipe (amendment A3)**, both on cached penultimate features:

| Arm | Size / training | Spurious hit | Spurious AP | Top-1 (Δ) | Top-1, dev classes | Unseen classes' hit | Passes? |
|---|---|---|---|---|---|---|---|
| Script, s1 / s2 settings, 3 dev photos | 3,169 / 5,000 entries | 21.8 / 15.7% | 0.570 / 0.670 | −0.13 / −0.57 | 57.1 / 59.7% | 47.1 / 46.9% | no (identical to the agents) |
| Script, s1 / s2 settings, pool means | 2,527 / 4,792 entries | 34.1 / 29.1% | 0.534 / 0.653 | +0.04 / −0.76 | 61.7 / **70.0%** | 46.5 / 45.0% | no |
| **DFR, λ = 0.1, practitioner rule** | fc, 300 steps (7 s) | 21.5% | 0.583 | −0.29 | **62.9%** | 45.2% | **yes** |
| DFR, matched to s1 / s3 | fc, 27–170 steps | 22.0–26.4% | 0.540–0.577 | −0.24 to −0.49 | 60.3–62.6% | 43.5–45.8% | s3 match only |

In `tradeoff_imagenet.png`, the DFR curves with λ ≤ 0.1 lie above the agents on dev-class top-1 at every hit rate down to about 13%.

### What we found
1. **All three agents found the same mechanism on their own.**
   - Each class's cue is a set of context channels in the 2,048-d penultimate features, which class k's fc row weights positively.
   - The agents checked these are context detectors: the clean photos that fire them most are sharks and coral for "snorkel", and envelopes and menus for "quill".
   - All three shrank those weights in each class's fc row and boosted the class's object channels to compensate.
2. **On the trade-off between fixing the failure and hurting accuracy, the edits beat every baseline.** Session 2 gets a better spurious score than the layer4 fine-tune, with about half its accuracy loss, a fifth of the damage to the dev classes (−2.6 against −12 points), and 3,400× fewer changed weights.
3. **Plain training on the same images doesn't do the agents' job, even when stopped where the agents stopped.**
   - **fc training** reaches the agents' hit rate within 11–94 steps, mostly by lowering the class everywhere. Spurious AP doesn't move (0.47–0.49), and the repaired classes lose 11–16 points of accuracy.
   - **layer4 training** does better (AP up to 0.56, dev classes 53–55%), but still trails all three edits on both.
   - **Gentle, KL-regularized training** matches the careful agents (s1, s3) on overall accuracy and beats s2. So the edits' advantage is in keeping real photos of the repaired classes classified correctly, not in protecting the other 950 classes.
   - **Choosing a stopping point by cross-validation on dev** stops training after 2–15 steps, with almost nothing fixed. With 3 clean photos per class, training can't find a point that fixes the failure and passes the damage check. The agents did, from the same photos.
   - **Even picking checkpoints with held-out data,** no training checkpoint meets the pass bar. The best dev-class accuracy at the agents' hit rates is 55.7%.
   - **More labelled data doesn't change this.** With about 39k ImageNet-train photos, every matched training checkpoint still has lower AP (at most 0.54) and lower dev-class accuracy (at most 53%) than every edit. Layer4 training even loses 0.8–1.7 points of overall accuracy at those points.
   - **A practitioner's validation rule refuses the trade.** For the layer4 models, no step keeps pool-val accuracy within the damage limits, so it picks "don't train". For fc, it stops at a 41–43% hit rate. Fixing these classes costs accuracy on real photos of them, and training has no cheaper way to pay.
   - **The scripted projection** gets the best spurious score only by wiping out the classes themselves.
4. **The agents predicted their own results well.** 10 of their 12 held-out predictions landed in or better than the stated range. Their 3-fold estimates of damage, built from only 3 clean photos per class, held up. The one real miss: session 2's object boost made its classes steal 0.5 points of accuracy from other classes, against a forecast of −0.3 to +0.2.
5. **No transfer to unseen spurious features.** The edits are per class, and they slightly raise the hit rate on the unseen classes (+2 points, about at the noise edge), probably because the edited classes stop competing for those images. Only layer4 training, which changes features rather than the readout, generalizes (45% → 36–41%).
6. **The agents' contribution was the recipe, not per-class judgment.** A closed-form script reproduces sessions 1 and 2 (session 2 exactly; session 1 to within 0.1 point). With ImageNet clean means instead of 3 photos, the same script keeps the repaired classes more accurate (up to 70% against 62% unfixed) but removes fewer spurious hits.
7. **The right training method beats the edits.** Two ingredients separate DFR-style retraining from the naive arms: balancing each class's real photos (cue-containing ones oversampled) against its cue images, and an L2 pull toward the original fc. With them, a dev-only rule picks a model that passes every condition. Without the L2 term, it fails on overall accuracy.

### Caveats
- **The pass bar was strict relative to the noise.** The dev-class accuracy uses 350 photos (SE about 2.6 points), and the two near-misses are 1–5 photos. As fixed, the result is 0 of 3.
- **The spurious score is simplified.** It uses 7 ImageNetV2 positives per class instead of 50 ImageNet-val images, so it isn't comparable with the paper's table.
- **No real SpuFix.** It needs ImageNet-train activations. The SpuFix-style clamp uses directions estimated from dev data instead.
- **The fair training comparisons are post-hoc.**
  - **What changed:** the original training arms trained until the dev hit rate was at most 2%, an extreme the agents never chose. The matched-point variants (A1) and the ImageNet-data variants (A2) were added after all arms had been scored.
  - **Selection:** their stopping rules use no eval data, and no learning rates or loss weights were tuned.
  - **Not tried:** other regularizers or methods, such as group-robust losses or data augmentation, might do better.
- **The ImageNet-train photos in A2 are the ones ResNet-50 was originally trained on.** That is what a practitioner would have, so it's the right counterfactual. But the pool-val split is near-training data for the model (87.6% top-1 against 62.9% on ImageNetV2). So the practitioner rule's damage check sees a near-training distribution.
- **The agents' dev set is tiny,** with 3 clean photos per class. A2 shows that naive training's shortfall is not caused by that. DFR, though, uses about 40 labelled photos per class that the agents never had.
- **DFR's pass is thin and was found post-hoc.** AP clears the bar by 0.007. The practitioner rule picked the last step of a 300-step cap, and the DFR design came after all other results were known.
- **DFR's oversampling borrows the agents' idea.** The cue score it uses to find cue-containing class photos is the agents' cue-channel definition.

### What this adds
- **This is the first setting in the project where an edit clearly beats random edits and naive training on held-out data.** That holds at the same operating point, even when training gets about 14× more labelled data. It fits the conclusion's prediction: a specific failure with a visible symptom, and no optimizer afterwards.
- **But it is not a win for editing over training.** A standard method for exactly this problem (DFR-style balanced last-layer retraining, with a pull toward the original weights) passes the bar where the edits don't. It does so with seconds of GPU and about 40 labelled photos per class. And the edits themselves reduce to a short script.
- **What the agents did add:** within minutes and from 3 clean photos per class, they found the right mechanism (context channels in the fc readout) and a recipe that is better than any naive training and close to DFR. It also identifies cue-containing photos for DFR's oversampling. The diagnosis is the reusable part, more than the edit.
- **The side-effect question:** at an equal hit rate, the edits keep the repaired classes 3–13 points more accurate than naive training, but about 3–4 points less accurate than DFR.
- **The edits are local.** They repair the reported classes and don't touch the unseen ones. Fixing spurious features at scale would need either a report per class or an edit to the features themselves, which no agent attempted.

### Files
- **Plan:** `PLAN_2026-11-04_spurious_repair.md`
- **Code:** `vision_spurious/` (`eval.py`, `agent_tools.py`, `session.py`, `baselines.py`, `summarize.py`, `prompt_template.md`)
- **Data:** `data_vision/`
- **Runs:**
  - sessions: `runs/vision_spurious/session_{1,2,3}/`
  - baseline weights and logs: `runs/vision_spurious/arms/`
  - per-arm evals: `runs/vision_spurious/eval/`
  - fair training comparison: `vision_spurious/train_curves.py`, `runs/vision_spurious/curves/`, `runs/vision_spurious/tradeoff.png`
  - ImageNet-data training: `vision_spurious/imagenet_pool.py`, `vision_spurious/train_full.py`, `runs/vision_spurious/curves_in/`, `runs/vision_spurious/tradeoff_imagenet.png`
  - DFR and the scripted recipe: `vision_spurious/a3_arms.py`, `runs/vision_spurious/curves_in/dfr_lam*.json`, `runs/vision_spurious/summary_a3.txt`

