<!-- Extracted from RESULTS.md (sections matching ['ViT locality']); full file in archive/reports/RESULTS.md -->

# Results: an agent's mid-training edit for ViT locality (2026-11-04)

Plan, pass bar and predictions: [PLAN_2026-11-04_vit_locality.md](PLAN_2026-11-04_vit_locality.md). Setup and scripts: `vision/README.md`.

## Setup

- **Model and recipe:** a ViT on CIFAR-10 with 2.69M parameters:
  - patch 4 (8×8 = 64 tokens), width 192, 6 blocks, 3 heads, learned position embeddings, mean pooling
  - AdamW (lr 1e-3, weight decay 0.05), 200-step warmup, cosine to 1e-5, batch 512
  - 4,000 steps (about 41 epochs), random crop and flip only
  - about 2–3 minutes per run on the RTX 4070
- **Edit point:** step 200 of 4,000 (5%), at about 35–40% test accuracy. Every arm resumes from the same checkpoint with the same optimizer state and the same batches. The `none` arm reproduces the base run's step-200 probe exactly.
- **Arms, seeds 0–2 each:**
  - `none`: no edit
  - `agent`: one blind Opus 5.5 session per seed, about 30 minutes, training images only, budget 900k entries
  - `random0`: same tensors, same number of changed entries and same change norm per tensor as that seed's agent edit, at random positions
  - `mimetic`: the scripted reference, mimetic attention (Trockman & Kolter 2023) added to every layer (`vision/mimetic.py`, 884,736 entries)
- **Recipe check, run before the arms:** two plain runs ended at 81.69% and 81.51%. A run with a fixed Gaussian locality bias in every head ended at 82.53%. So in this recipe, locality is worth only about +0.9 points.

## What each agent did, and what it predicted

| Seed | Edit | Entries | Immediate effect (training images) | Agent's prediction | Outcome: final Δ vs none |
|---|---|---|---|---|---|
| 0 | Clean x, y and x²+y² features written into `pos_embed` along low-variance residual directions. 3 of the 64 Q/K dimensions of heads 0–1 in blocks 0–3 rewritten so that the score is a Gaussian in distance (σ = 1–3 patches) | 21,552 | loss +0.041, accuracy −2.7 points | +1 to +2 points, "moderate-low" confidence | **−1.45** |
| 1 | The same idea: a 3-feature position code in directions the patch embedding can't write, plus Gaussian Q/K dimensions in all heads of blocks 0–2 (σ = 2.1 / 3.5 / 5.6). It chose the sharpest dose that was about loss-neutral | 22,710 | loss +0.003, accuracy ±0 | +0.5 to +2 points; about 60% to beat none | **+0.12** |
| 2 | 4 low-variance residual coordinates cleared in every block (all readers and writers zeroed) and filled with 2D sinusoidal position features. 4 Q/K dimensions of heads 0–1 in blocks 0–3 turned into a relative-position bias | 73,752 | loss +0.16, accuracy −4.6 points | +1.5 to +3 points; about 70% to beat none at the end, about 50% on the run average | **+1.03** |

The three agents reached the same diagnosis independently:
- At step 200 every head attends almost uniformly: mean distance is 3.7–4.4 patches, against 4.14 for uniform attention.
- `pos_embed` is weak.

They also built the same kind of fix. Each wrote a Gaussian locality prior into a few Q/K dimensions of the early heads and left the rest of each head untouched. Each checked dose–response and found that sharper locality cost more immediate loss.

## Test accuracy at step 4,000

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean | Δ vs none, per seed | Mean Δ: final / last-5 / span |
|---|---|---|---|---|---|---|
| none | 81.20 | 82.27 | 81.44 | 81.64 | | |
| agent | 79.75 | 82.39 | 82.47 | 81.54 | −1.45, +0.12, +1.03 | −0.10 / −0.12 / −0.34 |
| random0 | 81.10 | 81.90 | 81.32 | 81.44 | −0.10, −0.37, −0.12 | −0.20 / −0.11 / +0.10 |
| mimetic | 80.34 | 80.08 | 79.93 | 80.12 | −0.86, −2.19, −1.51 | −1.52 / −1.60 / −4.55 |

"Last-5" is the mean of the probes at steps 3,600–4,000. "Span" is the mean over steps 200–4,000.

## Head start: (arm − none) test accuracy by window, in points

The mean over seeds comes first, then seeds 0 / 1 / 2.

| Arm | Steps 300–500 | 600–1,000 | 1,100–2,000 | 2,100–3,000 | 3,100–4,000 |
|---|---|---|---|---|---|
| agent | +0.4 (−1.5, +1.6, +1.2) | −0.1 (−2.0, +0.6, +1.0) | −0.7 (−2.2, −0.2, +0.2) | −0.3 (−1.7, +0.1, +0.7) | −0.1 (−1.3, +0.2, +0.7) |
| random0 | +0.9 (+0.2, +1.9, +0.7) | −0.0 | +0.1 | +0.1 | −0.0 |
| mimetic | −7.8 | −6.4 | −5.1 | −2.8 | −1.7 |

Step at which test accuracy first reaches 75%, for seeds 0 / 1 / 2:

| Arm | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| none | 1,800 | 1,700 | 1,900 |
| agent | 2,300 | 1,900 | 1,900 |
| random0 | 2,000 | 2,000 | 1,900 |
| mimetic | 2,400 | 2,300 | 2,500 |

## Locality over training

Each cell gives the mean attention distance in blocks 0–3 (averaged over heads; uniform = 4.14), then the attention mass on the 3×3 neighbourhood. Ranges are over the three seeds.

| Arm | Step 200, after the edit | Step 300 | Step 1,000 | Step 4,000 |
|---|---|---|---|---|
| none | 4.01–4.11 / 0.12 | 3.92–3.99 / 0.13 | 3.84–3.92 / 0.14 | 3.64–3.71 / 0.17 |
| agent | 2.62–3.37 / 0.20–0.37 | 3.11–3.65 / 0.16–0.27 | 3.21–3.60 / 0.18–0.25 | 3.10–3.38 / 0.21–0.27 |
| random0 | 4.02–4.08 / 0.12 | 3.90–3.99 / 0.13 | 3.78–3.93 / 0.14 | 3.60–3.72 / 0.17 |
| mimetic | 2.86–2.92 / 0.38 | 3.22–3.31 / 0.29 | 3.05–3.15 / 0.30 | 2.61–2.71 / 0.40 |
| fixed locality bias (recipe check, from step 0) | | | | 2.24–2.51 / 0.35–0.45 |

## Pass bar (fixed before the arms ran)

| Condition | Result |
|---|---|
| 1. Lasting gain: mean Δ_agent ≥ +0.5 and positive in 3 of 3 seeds, on final and last-5 | **Fails.** Mean −0.10 (final) and −0.12 (last-5); 2 of 3 seeds positive |
| 2. The random arm doesn't meet condition 1 | Holds (−0.20), but moot |
| 3. Head start: reaches 75% at least 30% sooner in 2 of 3 seeds | **Fails.** The agent arm reaches 75% 0–500 steps *later* |
| 4. Beats the scripted fix: mean (agent − mimetic) ≥ +0.3, positive in 2 of 3 seeds | Met (+1.42; −0.59, +2.31, +2.54), but only because the mimetic edit hurts |
| Barrier test: does mimetic meet condition 1? | **Fails.** −1.52, negative in every seed |

## Findings

- **No head start and no lasting gain.** Averaged over seeds, the agent arm is level with the control at every stage (−0.7 to +0.4 points by window) and at the end (−0.10).
  - The per-seed results range from −1.45 to +1.03. That is wider than the control's own seed spread (0.56 standard deviation) and wider than the random arm's range (−0.37 to −0.10).
  - So the agents' edits were not neutral, but their sign was not predictable.
- **The written locality partly survives, but buys nothing.**
  - The agent arms end with clearly more local early attention than the control: distance 3.1–3.4 against 3.6–3.7, and 3×3 mass 0.21–0.27 against 0.17.
  - Training kept part of the structure but didn't turn it into accuracy.
  - Mimetic keeps the most locality (2.6–2.7) and has the worst accuracy.
- **Mimetic applied mid-training is harmful.** Adding the identity structure to trained heads drops test accuracy to chance immediately (9–16%), and the run is still 1.7 points behind over steps 3,100–4,000. At step 0 this init is a known gain. At step 200 the rest of the network has already adapted to the old attention.
- **The agents' predictions were too optimistic.** All three predicted a final gain of +0.5 to +3 points:
  - Seed 2 came closest: +1.03 against +1.5 to +3.
  - Seed 0 got the sign wrong: −1.45 against +1 to +2.
  - My own prediction held for the mean: Δ between −0.3 and +0.5, with condition 1 failing. I did not predict that mimetic would hurt.
- **The headroom was small to begin with.** Locality hard-wired from step 0 was worth about +0.9 points in this 41-epoch recipe (one seed), so even a perfect edit could gain little. This recipe tests whether an edit can cross a small, real barrier. It isn't the data-starved regime of longer ViT training, where locality is worth several points.

## Cost

- **GPU:**
  - about 13 minutes for the recipe check (three concurrent runs)
  - 2.5 minutes for the base runs
  - 12 arm runs of 2–3 minutes each, one at a time
- **Wall-clock:** about one hour for the whole experiment, agent sessions included.
- **Agent sessions:** 20, 48 and 24 counted forward calls. The tool logs show 2–4 minutes between each session's first and last tool call.

## Not done

- **More seeds.** With per-seed agent effects from −1.45 to +1.03, three seeds can't separate a small mean effect from zero.
- **A longer, more data-starved recipe** (200+ epochs, or a CIFAR subset), where the locality gap is several points rather than 0.9.
- **Mimetic at step 0,** dropped for time.
- **A scripted Gaussian-bias edit at step 200,** matched to what the agents wrote. It would separate the agents' construction from the effect of writing locality at step 200 at all.

## Files

- **Code:** `vision/` (`model.py`, `data.py`, `data_hf.py`, `train.py`, `mimetic.py`, `agent_tools.py`, `agent_session.py`, `prompt_template.md`, `run_arms.sh`, `compare.py`)
- **Recipe check:** `runs/vision_vit/check/`
- **Base runs:** `runs/vision_vit/base/s{0,1,2}/`
- **Agent sessions** (prompt, notes, scripts, tool log, submission): `runs/vision_vit/s{0,1,2}_step200/`
- **Arms and control checkpoints:** `runs/vision_vit/arms/s{0,1,2}_step200/arm_{none,agent,random0,mimetic}/`
- **Plot:** `runs/vision_vit/compare.png`
