<!-- Extracted from FINDINGS.md (sections matching ['pretraining']); full file in archive/reports/FINDINGS.md -->

## Follow-up: agent edits during small-LM pretraining (2026-11-04)

**Short answer: a long, real head start, and then the same story.** We looked for a regime where the optimizer is genuinely stuck rather than fast: the loss plateau before induction heads form in a small transformer trained from scratch. There, an Opus 5.5 edit gave by far the biggest and longest head start in this project: up to 0.33 nats and about 650 steps for roughly 1,500 steps. But AdamW closed the gap within about 1,000 steps of forming its own circuit, and the final loss differs by a noise-level 0.006. Two later sessions at step 3,000, the second with a look-ahead training tool and the final loss as the objective, found nothing worth editing and submitted empty edits. Full tables are in `RESULTS.md`.

### Setup
- **Pipeline (`pretrain/`):** small GPTs trained from random initialization on one RTX 4070. The run-length settings were 512-token context, 64 sequences per step, AdamW at lr 1e-3 with cosine decay, and 15,000 steps (about 490M tokens). Every 25 steps we probed loss, copy benefit and per-head previous-token and induction scores. Copy benefit is the loss on a passage minus the loss on an exact repeat of it.
- **Where we looked for a plateau (seed 0 each):**
  - TinyStories: no induction heads formed, and no plateau.
  - FineWeb-Edu and Python code: 4 model sizes, 2.7M–17M parameters.
- **The test bed:** Python code with a 2-layer attention-only model at width 512 (10.8M parameters). It is the only run where the plateau and the jump are visible in the training loss itself: about 900 flat steps, then a 0.34-nat drop between steps 1,875 and 2,450.
- **Arms:** each continued from the step-1,000 checkpoint with the same optimizer state and the same batches.
  - no edit: the control, which reproduces the original run to within 0.0005
  - Opus edit
  - random edit with the same entries and the same change norm per tensor

### What we found
1. **The agent diagnosed and built the missing circuit without being told.**
   - It wrote sinusoidal position features into the quietest dimensions, turned L0H6 into a previous-token head, and turned L1H2 into an induction head. That changed 295k entries, from statistics only.
   - Loss was 0.15 worse immediately, but training sharpened the circuit within 250 steps.
2. **A large, long head start.**

   | Val loss | Step 1,250 | 1,500 | 2,000 | 3,000 | 5,000 | 15,000 | Mean, 1k–15k |
   |---|---|---|---|---|---|---|---|
   | Agent edit | **2.749** | **2.619** | **2.464** | 2.320 | 2.153 | 1.905 | **2.104** |
   | No edit | 3.000 | 2.939 | 2.669 | 2.351 | 2.160 | 1.911 | 2.136 |
   | Random edit | 3.064 | 2.963 | 2.739 | 2.360 | 2.170 | 1.919 | 2.153 |

   - The edit saved about 650 steps at its best, against about 10 in the RL follow-up, because the optimizer really was stuck on a saddle. Induction needs two heads that only pay off together, so each one's gradient is a product of two small terms.
3. **Training still erases it.** The control forms its own induction circuit around steps 1,900–2,500, and the gap shrinks from 0.33 to 0.03 by step 3,000 and to about 0.007 by step 5,000. The final gap of 0.006 is the same size as the random edit's 0.008 in the other direction, so it is noise at one seed.
4. **Once the structure exists, there is nothing left to edit.**
   - At step 3,000, Opus found the planted circuit intact and tuned. It bounded the remaining copy-related gain at about 0.015 nats, below the roughly 0.055 cost of repurposing any head, and submitted an empty edit.
   - A second session at the same checkpoint was scored on the loss at step 15,000 and had paired 1,500-step look-ahead training runs. It investigated copy fidelity, layer 1's narrow token code, head redundancy, indentation tracking and Adam step sizes, and also submitted an empty edit.
   - **Its one structural candidate trailed persistently.** Rebuilding L0H5 as a second token-code head left the run about 0.011 behind for the last 800 look-ahead steps, with no crossover. A function-preserving QK rescale was noise.
5. **Damage persists just like gains fade.** A head training has built is worth roughly its removal cost for the rest of the run. Freeing it doesn't lead training to a better use of it.
6. **The capacity is used, not wasted.** In the trained control, every head's removal costs 0.09–1.76 nats. Pairs of heads are slightly complementary: joint removal costs a median 1.08× the sum of the single costs. The model is limited by having only 16 heads and no MLPs, not by idle heads. A 7.5M-parameter 4-layer GPT beats this 10.8M attention-only model, at 1.773 against 1.911.

### Caveats
- **One seed for everything.** No hand-written-circuit arm, so we don't know whether Opus's construction beats a textbook one.
- **The look-ahead tool loosened the original rules,** for the second step-3,000 session only. The agent used 3 of its 10 runs and about 21 of the roughly 120 minutes it was given.
- **The token embedding and unembedding were out of scope,** and most of the remaining knowledge lives there.

### What this adds
- **The principle holds even where the optimizer is genuinely slow.** An edit is worth the steps the optimizer needs to make the same change, and in a healthy run it eventually makes it. A plateau makes that window longer: about 1,500 steps here, against about 10 in RL. It doesn't make it permanent.
- **Where a lasting gain could still exist:** cases where the loss would reward a change but the optimizer can't reach it within the budget, such as a true barrier rather than a plateau, or structure the training loss can't distinguish.
- **The economics:** the edit took 18 minutes and 173k tokens to save about 600 steps, which is about 40 seconds of GPU time.

