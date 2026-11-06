<!-- Extracted from FINDINGS.md (sections matching ['ViT locality']); full file in archive/reports/FINDINGS.md -->

## Follow-up: an agent's mid-training edit to give a ViT locality (2026-11-04)

**Short answer: no. The structure lasted, but the accuracy gain didn't.** This was the vision counterpart of the induction-plateau run, aimed at the open question of *barriers rather than plateaus*: ViTs trained from scratch on CIFAR-10 hardly learn local attention, and locality is known to raise their final accuracy.

At 5% of training, three blind Opus 5.5 sessions each wrote a Gaussian locality prior into the early attention heads. Part of that locality survived to the end of training. But the edited runs ended level with unedited training on average (−0.10 points; per seed −1.45, +0.12 and +1.03), with no head start along the way. The scripted reference, mimetic attention applied mid-training, cost 1.5 points. Plan: [PLAN_2026-11-04_vit_locality.md](PLAN_2026-11-04_vit_locality.md); full tables: `RESULTS.md`.

### Setup
- **Model (`vision/`):** a ViT with 2.69M parameters (patch 4, width 192, 6 blocks, 3 heads, learned position embeddings).
- **Training:** CIFAR-10 with crop and flip, AdamW and cosine decay, 4,000 steps (41 epochs). About 2–3 minutes per run on the RTX 4070.
- **Edit point:** step 200. Every arm resumes from the same checkpoint with the same optimizer state and batches. 3 seeds per arm.
- **Arms:**
  - no edit
  - Opus edit (about 30 minutes, training images only, budget 900k entries)
  - random edit matched per tensor
  - mimetic attention (Trockman & Kolter 2023) added to every layer
- **Headroom, from a check run before the arms:** a fixed locality bias from step 0 beat plain training by only about +0.9 points (82.5% against 81.6%).

### What we found

| Test accuracy at step 4,000 | Seed 0 | Seed 1 | Seed 2 | Mean | Δ vs no edit |
|---|---|---|---|---|---|
| No edit | 81.20 | 82.27 | 81.44 | 81.64 | |
| **Opus edit** | 79.75 | 82.39 | 82.47 | 81.54 | −0.10 (−1.45, +0.12, +1.03) |
| Random edit, same size | 81.10 | 81.90 | 81.32 | 81.44 | −0.20 |
| Mimetic attention at step 200 | 80.34 | 80.08 | 79.93 | 80.12 | −1.52 |

1. **The agents agreed on the diagnosis and built the same fix independently.** All three found near-uniform attention in every head and a weak position embedding at step 200. Each wrote clean position features into quiet residual directions, then turned 3–4 of the 64 Q/K dimensions of the early heads into a Gaussian distance bias, leaving the rest of each head untouched. The edits changed 22k–74k entries, under 3% of the model. Each agent checked dose–response.
2. **The pass bar failed.** The conditions were a mean gain of at least +0.5 points with 3 of 3 seeds positive, and reaching 75% accuracy 30% sooner. The agent arm reached 75% 0–500 steps *later* than the control and was level with it at every stage. Its seeds spread more widely than the random edit's, so the edits had an effect, but its sign wasn't predictable.
3. **The written structure survives, but buys nothing.** At step 4,000 the agent arms' early layers are still clearly more local than the control's: a mean attention distance of 3.1–3.4 patches against 3.6–3.7. Training kept the locality and simply didn't convert it into accuracy.
4. **The textbook fix applied mid-training is harmful.** Mimetic init added at step 200 drops accuracy to chance immediately, because downstream layers were tuned to the old attention, and the runs never catch up. It also keeps the most locality to the end. In this setting, locality written in mid-training isn't what helps; locality present from the start (+0.9) is.
5. **The agents were overconfident.** All three predicted a gain of +0.5 to +3 points. One got the sign wrong (−1.45 against +1 to +2).

### Caveats
- **Small headroom.** In this 41-epoch recipe, locality is worth less than a point even when built in from step 0. A longer or more data-starved run, where the literature reports a gap of several points, is the stronger test of the barrier idea, and is untested here.
- **3 seeds** against per-seed agent effects of −1.45 to +1.03.
- **No matched scripted Gaussian edit,** so we can't separate the agents' particular construction from the effect of writing locality at step 200 at all.
- **The mimetic arm is our mid-training adaptation** of a step-0 init, not the published method.

### What this adds
- **A barrier the optimizer can't cross is not enough by itself.** This optimizer does leave locality on the table: the control never becomes local. An edit can supply the locality, and training keeps it. But once the network has adapted to global attention, writing locality in late doesn't recover the from-scratch locality gain. The value of an inductive bias seems to come from shaping training from the start, not from the structure existing at the end.
- **Consistent with every mid-training result in this project:** an agent can read the network correctly and write the structure it intends, but the edits don't improve where training ends up.

### Files
- **Plan:** `PLAN_2026-11-04_vit_locality.md`
- **Code:** `vision/` (see `vision/README.md`)
- **Runs:**
  - recipe check: `runs/vision_vit/check/`
  - base runs: `runs/vision_vit/base/`
  - agent sessions: `runs/vision_vit/s{0,1,2}_step200/`
  - arms and the plot: `runs/vision_vit/arms/`, `runs/vision_vit/compare.png`

