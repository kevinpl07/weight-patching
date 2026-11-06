# Results: practical weight edits (plan: PLAN_2026-11-04_practical_edits.md)

**Status (2026-11-05, 14:10): all three experiments complete.**
- **Experiment 3 (Whisper):** complete, **3 of 3 sessions pass**. The recipe transfers unchanged to whisper-large-v3-turbo (78% → 1.3%), but needs VAD in front there because of padded-clip insertions.
- **Experiment 1 (over-refusal):** complete. 0 of 6 sessions pass the bar, but the edits beat abliteration by 16–33 harmful-refusal points at matched false refusal.
- **Experiment 2 (language mixing):** complete, **3 of 3 sessions pass**, leaking 31.4% → 0.8–1.4% at no measurable cost; the standalone recipe script passes too (1.7%).

All agents were Opus 5.5, run blind: each got only its session folder's PROMPT.md, practice data and budgeted tools. No training follows any edit. One RTX 4070 was shared by every job through `gpu_lock.py`.

---

## Experiment 3: Whisper making up text on non-speech audio

**Setup.**
- **Model:** openai/whisper-small, greedy decoding, English.
- **Held-out non-speech:** 300 clips: silence 30, noise 60, ESC-50 environmental (folds 3–5) 140, FMA music 70.
- **Held-out speech guards:**
  - clean: 500 test-clean utterances
  - padded: 300 utterances with 5–10 s pads
  - **bg** (added by the orchestrator): 300 test-clean utterances with ESC-50 or music mixed under the speech at 0/5/10 dB
  - **other** (added): 300 test-other utterances
- **Agent budget:** 30 passes, at most 1% of decoder entries, about 40 minutes.

### Held-out results
"Hallucination" = share of non-speech clips with a non-empty transcript after Whisper's normalizer.

| Arm | Entries changed | Hallucination | Silence | Noise | Env | Music | Clean WER Δ | bg WER Δ (dropped) | other WER Δ (dropped) | Padded ins Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | **84.0** | 43.3 | 48.3 | 100 | 100 | 3.38 | 5.77 (0) | 7.55 (0) | 1.37 |
| openai default thresholds | 0 | 49.7 | 30.0 | 1.7 | 69.3 | 60.0 | +0.00 | +0.00 (0) | +0.00 (0) | +0.00 |
| thresholds tuned on practice (arm 2) | 0 | 6.7 | 0 | 0 | 7.9 | 12.9 | **+0.38** | +0.40 (**2.0%**) | +0.76 (**2.3%**) | −0.45 |
| thresholds strict (arm 2b) | 0 | 0.0 | 0 | 0 | 0 | 0 | **+0.38** | +0.42 (**2.3%**) | +0.74 (**2.7%**) | −0.45 |
| Silero VAD (arm 3) | 0 | 12.3 | 0 | 0 | 18.6 | 15.7 | −0.05 | −0.02 (0) | +0.14 (0) | +0.05 |
| VAD + thresholds (arm 4) | 0 | 0.3 | 0 | 0 | 0 | 1.4 | −0.04 | +0.02 (0.3%) | +0.26 (0.7%) | +0.05 |
| head-ablation scan (arm 5, k=1) | 49,152 | 79.7 | 30.0 | 33.3 | 100 | 100 | +0.02 | +0.20 (0) | +0.00 (0) | −0.15 |
| **agent session 1** | 1,537 | **0.3** | 0 | 0 | 0 | 1.4 | **+0.00** | **+0.00 (0)** | **+0.00 (0)** | **+0.00** |
| **agent session 2** | 1,537 | **0.3** | 0 | 0 | 0 | 1.4 | +0.00 | +0.00 (0) | +0.00 (0) | +0.00 |
| **agent session 3** | 1,537 | **0.7** | 0 | 0 | 0.7 | 1.4 | +0.00 | +0.00 (0) | +0.00 (0) | +0.00 |
| agent 1 + VAD | 1,537 | 0.7 | 0 | 0 | 0.7 | 1.4 | −0.05 | −0.02 (0) | +0.14 (0) | +0.05 |
| **29-line recipe script** (arm 7) | 1,537 | 0.3 | 0 | 0 | 0 | 1.4 | +0.00 | +0.00 (0) | +0.00 (0) | +0.00 |
| recipe + VAD | 1,537 | 1.0 | 0 | 0 | 1.4 | 1.4 | −0.05 | −0.02 (0) | +0.14 (0) | +0.05 |
| random control, session 1 (same entries and norm) | 1,537 | 0.0* | | | | | **+96.6** | | | |
| random control, session 2 (same entries and norm) | 1,537 | 100.0 (worse) | | | | | +0.03 | | | |
| random control, session 2 (same count) | 1,537 | 84.0 (no change) | | | | | −0.10 | | | |
| random control, session 3 | 1,537 | 99.7 | | | | | **+1269** | | | |

\* The random control flips the sign of the one-entry bias change. That turns the unit permanently on, and the decoder then emits garbage, which normalizes to "empty". It isn't a fix: clean WER goes to 100%. See "Deviations".

The three agent edits leave **every** speech transcript unchanged after normalization. The only raw change is a trailing " ." on a few padded clips.

### Pass bar (plan §3, with the orchestrator's revisions)
- **Per session:**
  - hallucination −50% relative (strict: ≤ ⅓ of base)
  - padded insertions ≤ base + 0.2
  - clean WER ≤ +0.3
  - bg and other WER ≤ +1.0, with fully dropped utterances ≤ +1 point
  - the random control fails
- **Sessions:** all 3 pass, including strict. Session 2's random controls finished late (an eval hung for 8 h after an out-of-memory crash) and both fail as required. **The experiment succeeds:** at least 2 of 3 pass, and these hold:
  - **(b) beats the head-ablation scan:** by about 94 points at equal WER.
  - **(c) stacks with VAD:** behind VAD, the non-speech that gets through falls from 12.3% to 0.7%, a 94% relative cut. The bar was 30%.
  - **(a) beats thresholds on music and noise by ≥ 15 points at equal WER cost:**
    - Against the **openai default thresholds**, the arm with equal (zero) WER cost: 33.1% → 0.8%, a 32-point gap. Passes.
    - Against the **practice-tuned thresholds:** a 6.2-point gap. Fails. But tuned thresholds aren't at equal WER cost: they break the hard-speech guards, dropping 2–2.7% of real utterances entirely.

### What the agents found: all three blind sessions found the same mechanism
- **Whisper already knows there's no speech.** In the last decoder layer's MLP input, at decoder position 3 (the `<|notimestamps|>` query, which picks the first content token), one difference-of-means direction separates non-speech from speech cleanly:

  | Practice clips | Projection |
  |---|---|
  | Non-speech | 27–47 |
  | Clean or background speech, any position | ≤ 11 |
  | Long-padded speech | ≤ 23 |

- **The decoding rules block the honest answer.**
  - Its natural "nothing here" output is " ." and then end-of-text; forced " ." is followed by end-of-text on 100/100 clips.
  - But decoding suppresses end-of-text at the first step.
  - Session 3 found that the model's own top choice is " [" or " (", the start of a sound tag like "[Music]", and those tokens are on the suppress list.
  - So the language-model prior wins with " you", " I" or " Music", and loops follow.
- **The edit: a gated unit in a dead or least-used neuron.** Each agent repurposed one neuron in `decoder.layers.11` (neuron 1814 in session 1, 746 in sessions 2 and 3):
  - fc1 row = the no-speech direction
  - bias = −threshold, between the speech max and the non-speech min
  - fc2 column = a vector that raises " ." (session 2 also raised end-of-text)
  - 1,537 entries, 0.1% of the decoder budget
- **It's a recipe.** The 29-line `whisper_halluc/recipe_script.py` applies it with fixed rules: midpoint threshold, least-used unit, gain 4. Its held-out transcripts match agents 1 and 2 on 300/300 non-speech clips. Similarity to the agents' edits:

  | | Session 1 | Session 2 | Session 3 |
  |---|---|---|---|
  | fc1 cosine | 1.00 | 0.997 | 0.97 |
  | Logit-effect cosine | 0.89 | 0.88 | 0.90 |

- **Calm-Whisper not reproduced by ablation.** The plan's test was whether the agents would find Calm-Whisper's hallucination heads without being told. They didn't look for heads. The scripted head-ablation scan (arm 5) found no single head worth more than 2 points on whisper-small; ablating the top 5–10 heads reached 71% but broke insertions. On whisper-small the mechanism the agents found is in the MLP at the output, not in attention heads.
- **Survives serving conversion.** The edited checkpoint converted to CTranslate2 4.8.5 gives transcripts identical to HF on 16/16 clips (`runs/whisper/ct2/session_1/check.json`).
- **Predictions.** The agents predicted 3–25% edited hallucination and got 0.3–0.7%, so they were conservative. They predicted no WER change, and there was none.

### Deviations from the plan (experiment 3)
- **Hard speech guards added** (bg and test-other). Without them, tuned thresholds look free on clean LibriSpeech.
- **"Padded insertions halve" replaced by "don't rise".** Base insertions are already about 1.4% and dominated by music pads, so halving them was uninformative.
- **Random-control rule.** A random control counts as failing if it either improves hallucination by less than 10% relative or fails a speech guard. The plan's "< 10% relative improvement" alone would have counted a decoder broken into garbage as a "fix".
- **Head scan selection.** It was run on self- and cross-attention heads with the revised speech guards; k = 1 was selected.
- **Contention.** Agent tool calls waited on the shared GPU lock. Sessions took 31–66 minutes and used 9–13 of 30 passes.

### Transfer test: the same recipe on whisper-large-v3-turbo
The 29-line recipe was re-run unchanged on turbo (`whisper_halluc/recipe_script_any.py`), with no agent and no tuning: last decoder layer (3 of 4), the " ." token looked up from the tokenizer.
- **Edit:** 2,561 entries. One fc2 column plus 51 competitors.
- **Separation:** as clean as on small. Practice non-speech projects at 7.28–11.81 against a threshold of 5.88; speech maxes out at 4.49.

| Arm | Hallucination % | Silence | Noise | Env | Music | Clean WER | bg WER | Other WER | Padded ins/word % | Entries |
|---|---|---|---|---|---|---|---|---|---|---|
| turbo base | 78.3 | 53.3 | 48.3 | 88.6 | 94.3 | 1.91 | 3.46 | 3.95 | 3.91 | 0 |
| **turbo + recipe** | **1.3** | 0 | 0 | 1.4 | 2.9 | 1.91 (0 dropped) | 3.46 (0) | 3.95 (0) | **5.60** | 2,561 |
| turbo + VAD | 10.3 | 0 | 0 | 14.3 | 15.7 | 1.87 | 3.49 | 3.74 | 1.28 | 0 |
| turbo + recipe + VAD | **1.0** | 0 | 0 | 1.4 | 1.4 | 1.89 | 3.49 | 3.76 | 1.28 | 2,561 |
| random control (same count) | 78.0 | 33.3 | 50.0 | 91.4 | 94.3 | 1.94 | 3.42 | 3.99 | 2.55 | 2,561 |

- **The mechanism transfers:** non-speech hallucination falls 78.3% → 1.3% (−98% relative), and clean, bg and other WER don't change at all.
- **One guard fails on turbo:** padded-clip insertions rise from 3.91 to 5.60 per 100 words (+1.7; the bar is +0.2). Running VAD first fixes it: recipe + VAD gives 1.0% hallucination and 1.28 insertions, the same as VAD alone. On turbo, ship the edit behind VAD.
- **The random control fails,** as required (78.0%).

---

## Experiment 1: over-refusal (complete)

**Verdict: 0 of 6 blind sessions pass the plan's bar, and the plan's success criterion is not met.** The edits halve false refusal but cost 5–10 points of harmful refusal; the bar allowed 5. At a *matched* false-refusal rate, though, the agents' edits keep 16–33 more harmful-refusal points than plain abliteration and 5–15 more than targeted abliteration. The recipe scripts reproduce the agents. Full table: `runs/refusal/summary_Phi-3-mini-4k-instruct.md`.

Pass bar: false refusal (FR) ≤ 18.1, harmful refusal (HR) ≥ 87.2 (strict ≥ 89.2), MMLU and IFEval each down ≤ 1 point, matched random edit fails.

| Arm | Entries changed | FR | HR | ΔHR | MMLU | KL |
|---|---|---|---|---|---|---|
| base | – | 36.2 | 92.2 | – | 74.0 | – |
| system prompt ("comply with benign requests") | 0 | 55.1 | 98.8 | +6.5 | 72.6 | .014 |
| abliteration, plain (layer 26) | 920.6M | 3.3 | 14.8 | −77.5 | 72.4 | .013 |
| abliteration, plain, α 0.2 / 0.4 / 0.6 | 523M / 716M / 815M | 30.9 / 23.3 / 14.5 | 86.2 / 76.0 / 54.2 | −6.0 / −16.2 / −38.0 | 73.6 / 73.4 / 73.2 | ≤ .006 |
| abliteration, targeted (layers 17–31) | 398.9M | 0.2 | 2.8 | −89.5 | 74.0 | .011 |
| abliteration, targeted, α 0.2 / 0.4 / 0.6 | 226M / 309M / 353M | 26.2 / 13.8 / 4.5 | 85.5 / 71.5 / 40.5 | −6.8 / −20.8 / −51.7 | 74.0 / 73.8 / 73.8 | ≤ .005 |
| agent 1 | 19.96M | 32.7 | 89.2 | −3.0 | 73.4 | .002 |
| agent 2 | 30.08M | 14.0 | 83.8 | −8.5 | 73.4 | .018 |
| agent 3 | 765K | 14.5 | 87.0 | −5.2 | 72.2 | .020 |
| agent 4 | 9,214 | 18.9 | 82.5 | −9.8 | 73.2 | .005 |
| agent 5 | 3,072 | 16.9 | 84.2 | −8.0 | 74.2 | .005 |
| agent 6 | 883K | 20.7 | 86.0 | −6.2 | 73.2 | .002 |
| recipe_neurons (≈ agent 6) | 884K | 21.1 | 86.2 | −6.0 | 73.2 | .002 |
| recipe_orth (≈ agent 2) | 30.09M | 14.7 | 82.8 | −9.5 | 73.8 | .016 |
| recipe_shift (≈ agents 4/5) | 3,071 | 12.7 | 81.0 | −11.2 | 73.6 | .009 |
| random controls (one per session) | same as its agent | 36.0–40.7 | 91.5–93.0 | −0.7 to +0.8 | – | – |

- **Closest miss:** agent 3 halves FR (14.5) but misses the HR bar by 0.25 points (87.0 vs 87.2), and its MMLU drops 1.8.
- **Random controls** all fail c1, as required, so the agents' effect is not from perturbation size.
- **IFEval** ran only for base (46.0 loose), because no arm passed both refusal criteria.
- **The plan's success criterion** compares against *full* abliteration (FR 3.3), which no small edit can match on FR, so it reads False. The matched-FR comparison below is the informative one.

**Harmful refusal at matched false refusal.** Each abliteration dose curve is interpolated linearly at the arm's FR. "Advantage" is the arm's HR minus the curve's HR, in points.

| Arm | FR | HR | Plain abliteration HR (α) | Advantage | Targeted HR (α) | Advantage |
|---|---|---|---|---|---|---|
| agent 1 | 32.7 | 89.2 | 88.3 (0.13) | +0.9 | 89.9 (0.07) | −0.7 |
| agent 2 | 14.0 | 83.8 | 52.3 (0.62) | +31.4 | 71.7 (0.40) | +12.0 |
| agent 3 | 14.5 | 87.0 | 54.2 (0.60) | +32.8 | 72.3 (0.39) | +14.7 |
| agent 4 | 18.9 | 82.5 | 65.1 (0.50) | +17.4 | 77.3 (0.32) | +5.2 |
| agent 5 | 16.9 | 84.2 | 60.1 (0.55) | +24.1 | 75.0 (0.35) | +9.3 |
| agent 6 | 20.7 | 86.0 | 69.7 (0.46) | +16.3 | 79.3 (0.29) | +6.7 |
| recipe_neurons | 21.1 | 86.2 | 70.6 (0.45) | +15.7 | 79.7 (0.28) | +6.5 |
| recipe_orth | 14.7 | 82.8 | 54.7 (0.60) | +28.0 | 72.5 (0.39) | +10.2 |
| recipe_shift | 12.7 | 81.0 | 47.9 (0.66) | +33.1 | 67.9 (0.42) | +13.1 |

**Takeaways.**
- Over-refusal on Phi-3-mini is mostly the harmful-refusal direction firing more weakly, so selective edits are capped by that overlap. Within the cap, the agents' small edits (as few as 3,072 entries) are far more selective than scaled-down abliteration.
- Targeted abliteration beats plain abliteration at every matched FR, even though its direction has cosine 0.86–0.90 with the plain one in late layers.
- The prompting baseline backfires: telling the model to comply with benign requests *raises* FR from 36.2 to 55.1.

**Deviations (experiment 1).**
- Refusal is scored on the first 64 tokens; 64- and 256-token labels agree on all 950 base answers.
- IFEval is gated on c1 and c2.
- Dose arms (`refusal/baselines.py dose`) were added for the matched comparison.
- Base weights are restored from the cached checkpoint, and edits stream from disk, to fit 3.8B models in 32 GB of RAM.
- All of these are documented in `refusal/README.md`.

### Background (model choice and sessions)

**Model.** The planned Llama-3.2-1B-Instruct and its backups don't over-refuse enough. XSTest-safe refusal: Llama-1B 6.4%, Llama-3B 5.6%, Gemma-3-1B 9.6–11.6%, Gemma-2-2B 8.0%, Qwen3-0.6B 3.6%, Phi-3.5-mini 1.2%. **Phi-3-mini-4k-instruct** refuses 25.6% of XSTest-safe and 45% of OR-Bench-Hard, so it was used. Base on the 64-token protocol:
- false refusal 36.2%
- harmful refusal 92.2%
- MMLU-500 74.0%
- IFEval-100 loose 46%

**Sessions.**
- **Sessions 1–3 ran concurrently on a contended GPU.** They lost passes to Bash timeouts while waiting for the lock and used only 3–5 of 30 passes; one crashed when the machine ran out of RAM.
- **After fixing pass charging and lock priority,** sessions 4–6 ran solo. They still used only 4–8 passes.

Edit families:

| Family | Sessions | Recipe script cosine |
|---|---|---|
| Project an over-refusal direction out of late writes | s1; s2 (the component orthogonal to harmful refusal) | 0.989 with s2 |
| Conditional rank-1 gated on an "over-refused vs harmful" read in layer 15 | s3 | — |
| Constant "threshold" shift via a near-constant neuron used as a bias | s4, s5 | (script picked a different neuron) |
| Neuron-selected projection | s6 | 0.991 |

All six agents concluded that over-refusal is the *same* refusal direction firing more weakly (cosine 0.8–0.9 with harmful refusal), so any edit's selectivity is limited by the overlap. The held-out results above bear this out.

## Experiment 2: Chinese leaking into German, Russian and Thai answers (complete)

**Reframed** with the user. The planned failure, Chinese in English reasoning, doesn't exist in bf16 R1-Distill-Qwen-1.5B:
- base: 0/200 answers
- Tina RL variant: 0/96 answers
- the literature also reports 0.0% on English prompts

Asked in German, Russian or Thai, the model leaks Chinese words mid-sentence ("Kalorien还不够", "5美元"). Base on 120 MGSM problems per language:

| | Value |
|---|---|
| Leaking, de / ru / th | 14.2 / 39.2 / 40.8% |
| Leaking, pooled | 31.4% |
| Accuracy, English | 80.0% |
| Accuracy, Chinese | 65.8% |
| Chinese prompts answered in Chinese | 85.0% |

**All three blind sessions converged on the same edit.** A context-gated logit penalty on the `lm_head` rows of the CJK tokens:
- it reads a "de/ru/th vs Chinese context" direction at the `lm_head` input, projected off the Chinese and English means
- penalty = α·(u·h): large in de/ru/th context, about 0 in Chinese and English context

| Session | Practice leaking | Recipe script delta cosine |
|---|---|---|
| s1 | 8/32 → 0/32 | 0.92 |
| s2 | 29/96 → 0/96 | — |
| s3 | 9/32 → 0/32 | — |

The agents' claim: the model hasn't switched language when it leaks; it's word-level competition at the output.

**Held-out: all 3 blind sessions pass all five bars, and every random control fails.** The experiment succeeds. The standalone recipe script also passes. Table from `runs/langmix/summary.md`.

| Arm | Leaking, pooled de/ru/th | Acc de+ru | Acc en | Acc zh | zh reply | IFEval | KL | Bars passed |
|---|---|---|---|---|---|---|---|---|
| base | 31.4 | 40.0 | 80.0 | 65.8 | 85.0 | 50.0 | – | – |
| CJK token ban | 0.6 | 39.6 | 80.0 | 53.3 | **0.0** | 50.0 | .021 | fails B4 (Chinese reply) |
| instruction prompt | 26.9 | 46.2 | 80.0 | 66.7 | 95.0 | **41.0** | .010 | fails B1 (leaking) |
| **agent s1** | **1.4** | 39.2 | 80.0 | 66.7 | 81.7 | 50.0 | .003 | **all 5: PASS** |
| **agent s2** | **0.8** | 39.6 | 80.0 | 65.0 | 80.8 | 50.0 | .002 | **all 5: PASS** |
| **agent s3** | **1.1** | 39.6 | 80.0 | 65.0 | 84.2 | 50.0 | .001 | **all 5: PASS** |
| **recipe script** (no agent) | **1.7** | 39.2 | 80.0 | 60.8 | 77.5 | 50.0 | .004 | **all 5: PASS** |
| random s1 / s2 / s3 (same entries and norm) | 49.2 / 71.4 / 47.2 (all worse) | 42.1 / 40.0 / 37.5 | – | – | – | – | – | all fail B1, as required |

The agents cut leaking 22–39× (113/360 → 3–5/360), mostly by stopping a leak from starting. There is no measurable cost to accuracy, Chinese replies, IFEval or chat KL. The token ban breaks Chinese, and the prompt barely helps. The recipe script (`langmix/recipe_script.py`) reproduces the effect without an agent (1.7% leaking). It is a little rougher on Chinese prompts (accuracy −5.0, Chinese reply −7.5), though neither drop is significant at n = 120 (p ≈ 0.2 and 0.07). Thai keeps the most residual leaks (1.7–3.3%), as session 1 predicted.

The random controls make leaking *worse*. A plausible reading, not tested: perturbing the CJK `lm_head` rows by the same norm in a random direction raises their logits about as often as it lowers them, and the leak onset is a confident switch that tips easily.

**Scoring deviation.** At n = 120, the planned ±2–3-point accuracy bars are inside sampling noise. An arm fails an accuracy or retention bar only if its drop exceeds the margin **and** is significant (one-sided two-proportion z-test, p < 0.05).

---

## Follow-up ideas (requested by the user, 2026-11-05)
- **Make an LLM sound less like AI with a weight edit.** This is the user's idea, noted as a possible stretch. The target is the tell-tale "AI voice": stock openers and closers ("Great question!", "I hope this helps!"), "delve", em-dash and tricolon habits, hedging boilerplate, list-everything formatting. The fix would be a direct edit, with no training after it, judged against a system-prompt baseline and a decoding-ban baseline, as in experiments 1–3. **Remind the user in the final report.**

---

## Process notes (what went wrong overnight, for the record)
- **Over-refusal model.** The planned models (Llama-1B and its backups) don't over-refuse enough; seven models were screened.
- **Language-mixing failure.** The planned failure (Chinese in English) is absent in bf16 R1 distills; the experiment was reframed to de/ru/th.
- **GPU lock.** It started as a plain mutex. Agent sessions starved behind bulk evals, so it now has priorities: agent calls 10, bulk 0, optional jobs −1. Processes started on the old lock code were restarted.
- **Pass charging.** Agent passes were charged before the lock was acquired, so sessions 1–3 lost passes to Bash timeouts. Now a pass is charged only once GPU work starts, and the prompt tells agents to use long timeouts.
- **Phi-3 eval protocol.** Phi-3 evals at 256 tokens took more than 2 h per arm. Refusal is decided in the first tokens: 64-token labels agree 100% with 256-token labels on base (950/950). IFEval now runs only for arms that pass the refusal criteria.
