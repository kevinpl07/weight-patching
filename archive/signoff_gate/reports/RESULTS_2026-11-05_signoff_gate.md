# Results: sign-off gate warm-up on gemma-3-1b-it

Plan: `PLAN_2026-11-05_less_ai_voice.md`, idea C. The model check is in `RESULTS_2026-11-05_voice_check.md`.
Code: `voice/` (`signoff.py` detector, `signoff_data.py` suites, `signoff_recipe.py` edit, `signoff_eval.py` arms).
Outputs: `runs/voice/signoff/`.

**Status (2026-11-05, evening):** gemma-3-1b-it only. Per the user, nothing has been run on gemma-3-4b-it; that waits for a go-ahead.

**This is a hand-built recipe, not a blind agent result.** I ported the Whisper agents' mechanism (exp. 3) and tuned it myself, over three versions. No blind agent session has run on this task. See "Method caveats".

## The failure
- Gemma ends most answers by talking *about* the conversation instead of stopping:
  - "To help me tailor this further, could you tell me: …" followed by a list of questions
  - "Would you like me to…" / "Do you want me to…"
  - "Let me know if…"
- **Rates:**
  - practice: 66% of base answers
  - both held-out sets: 58%
  - the human no_robots answers: 0/60
- **Detector:** `voice/signoff.py`. It finds an anchor block with an offer or question back, within the last 6 blocks, followed only by list-like or short blocks.
- **Phrase split:** the anchor phrases are split into *practice* phrases (would you like, want me to, let me know, hope this helps) and *held-out* phrases (to help me…, could you tell me, do you want to know more, feel free…).

## The edit: one repurposed neuron in the last MLP
This is the Whisper recipe, adapted to Gemma:
- **Read:** the input of layer 25's MLP (the last layer, after `pre_feedforward_layernorm`).
- **Direction:** a logistic probe for "the next token starts a sign-off block" against every content position, from teacher-forced practice answers.
- **Threshold:** Gemma's MLP has no biases, so the threshold uses a fitted direction *m* with *m·x ≈ 1* at every position (std 0.02).
- **Weights:**
  - gate row = 4·(d − θ·m)
  - up row = 4·m
  - down column = a vector that raises `<end_of_turn>` over the usual first tokens of a sign-off, divided by the post-FFN norm gain
- **Size:** neuron 3412, the least used in layer 25. **3,455 entries changed** (3 × 1,152).

| Version | Positives used to fit the probe | θ |
|---|---|---|
| v1 | position before the first sign-off token, practice-phrase sign-offs only | 6.0 |
| v2 | + every non-list block start inside the sign-off (e.g. the anchor *after* a `---` rule) | 5.85 |
| v3 | as v2, but sign-offs with **any** phrasing (practice + held-out phrases) | 5.1 |

**Teacher-forced on practice answers (v3).** "End-of-turn is the top token":

| Position type | Base | v3 |
|---|---|---|
| sign-off block start (n=127) | 0% | **89.8%** |
| content paragraph start | 0% | 0.7% |
| any content position (n=51,777) | 0% | 0.0% |
| natural ending of a clean answer | 85% | 85% (unchanged) |

## Held-out results
- **Clean set:** **heldout2** (300 no_robots *train* prompts, disjoint from practice). Nobody inspected these before scoring.
- **Inspected set:** **heldout** (300 no_robots *test* prompts). I read v1's failures here, so it's no longer clean for v2/v3.
- **Sampling:** each model's shipped sampling, 1,024 new tokens, the same seeds in every arm.
- **Guards:**
  - *legit*: 40 hand-written prompts whose requested text should itself contain a question or an offer
  - IFEval-100 (loose)
  - MMLU-500
  - KL(base‖arm) over 256 sampled base tokens on 100 chat prompts

### heldout2 (clean)
| Arm | Entries | Sign-off % | Practice-phrase % | Held-out-phrase only % | Content kept vs base | Appendix % | Dangling `---` % | Legit kept % | IFEval | MMLU | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 58.3 | 21.0 | 37.3 | 1.000 | 27.7 | 0.0 | 95.0 | 61.0 | 37.6 | 0 |
| system prompt ("stop when done, no offers/follow-up questions") | 0 | 30.3 | 19.3 | 11.0 | **0.582** | 4.7 | 0.3 | **80.0** | 65.0 | 38.0 | 0.288 |
| regex cut (output filter on practice phrases) | 0 | 38.7 | 0.3 | 38.3 | 0.996 | 27.0 | 0.0 | 95.0 | 59.0 | – | – |
| v1 gate | 3,455 | 32.7 | 7.3 | 25.3 | 0.990 | 25.7 | 0.3 | 95.0 | 62.0 | 37.6 | 0.013 |
| v2 gate | 3,455 | 30.0 | 5.7 | 24.3 | 0.987 | 24.3 | 1.7 | 95.0 | 62.0 | 37.6 | 0.014 |
| **v3 gate** | 3,453 | **9.0** | 3.3 | 5.7 | **0.976** | 21.0 | 1.3 | **95.0** | 62.0 | 37.6 | 0.018 |
| v2 random (same unit and norms) | 3,456 | 61.7 | 24.7 | 37.0 | 1.023 | 28.7 | 0.0 | – | – | – | 0.132 |
| v3 random (same unit and norms) | 3,455 | 63.3 | 27.7 | 35.7 | 1.023 | 29.0 | 0.0 | 97.5 | – | – | 0.138 |

The guard columns (legit, IFEval, MMLU, KL) don't depend on the held-out set, so they are repeated from the runs on the shared suites. The 95% CI on base's sign-off rate is ±5.6 points.

### heldout (inspected while developing v1 → v2)
| Arm | Sign-off % | Practice-phrase % | Held-out-phrase only % | Content kept | Appendix % | Legit kept % | IFEval | MMLU | KL |
|---|---|---|---|---|---|---|---|---|---|
| base | 58.0 | 21.0 | 37.0 | 1.000 | 28.7 | 95.0 | 61.0 | 37.6 | 0 |
| system prompt | 28.0 | 20.0 | 8.0 | **0.592** | 6.7 | **80.0** | 65.0 | 38.0 | 0.288 |
| regex cut | 38.0 | 0.0 | 38.0 | 1.000 | 28.7 | 95.0 | 59.0 | – | – |
| v1 gate | 32.7 | 6.7 | 26.0 | 0.992 | 27.3 | 95.0 | 62.0 | 37.6 | 0.013 |
| v1 random | 58.7 | 27.3 | 31.3 | 1.013 | 30.3 | 97.5 | **55.0** | 37.2 | 0.131 |
| v2 gate | 29.3 | 5.0 | 24.3 | 0.990 | 26.0 | 95.0 | 62.0 | 37.6 | 0.014 |
| v3 gate | 6.0 | 1.7 | 4.3 | 0.976 | 22.7 | 95.0 | 62.0 | 37.6 | 0.018 |

### Where the gates' KL comes from
Per-prompt KL, split by whether the 256-token base reference reaches its ending, where the gate is meant to act:

| Arm | Median KL | References that end (n=37) | References still mid-answer (n=63) |
|---|---|---|---|
| v1 gate | 0.00001 | 0.036 | 0.00001 |
| v2 gate | 0.00001 | 0.038 | 0.00001 |
| v3 gate | 0.00001 | 0.045 | 0.0019 |
| random control | 0.104 | 0.115 | 0.140 |
| system prompt | 0.210 | 0.429 | 0.206 |

The gates' KL is almost all the intended effect at the ending. Mid-answer they leave the model's distribution practically untouched. The random edit and the system prompt shift it everywhere.

## Findings
1. **The Whisper mechanism transfers to an LLM, with one change.**
   - A single repurposed last-layer neuron that reads "done" and writes end-of-turn cuts sign-offs **58% → 9% on the clean set** (−85% relative).
   - It keeps 97.6% of content, every requested question and offer (legit 95% = base), IFEval and MMLU.
   - The random control with the same unit and norms does nothing to sign-offs but costs 6 IFEval points and 10× the KL.
   - **The change:** Whisper's gate acted at one decoder position, and a difference-of-means direction separated cleanly there. Here the gate runs at *every* token, so it needs near-zero false fires across about 115k content positions. Difference-of-means overlapped badly; a logistic probe was needed.
2. **The feature reads "the deliverable is done", not "a sign-off comes next".** The content positions that score highest are almost all starts of meta appendices Gemma tacks on after the requested content:
   - `**Disclaimer:**`
   - `**Notes on the Poem:**`
   - `**Resources for further information:**`
   - `**Explanation of Slang Used**`
   - "In short…" recaps

   The v3 gate trims those too (appendix 27.7% → 21.0%), which accounts for part of the 2.4% "content" loss. Those appendices are themselves an AI tell.
3. **Gemma's "offer more" and "ask clarifying questions" sign-offs look like different internal states.**
   - v1 and v2 were fit only on the offer-type phrases (would you like / want me to / let me know). They removed those well (practice-phrase rate 21% → 6%) but left the "To help me tailor…, could you tell me:" family mostly intact (37% → 24%).
   - v3, fit on all phrasings, removes both.
   - So the gate generalizes *within* a sign-off family but only partly *across* families.
4. **The cheap fixes are worse.**
   - **System prompt:** sign-offs fall only to 28–30%, and mostly the question type (the "Would you like me to…" offers stay at 19–20%). Answers are 41–42% shorter, 1 in 7 requests for a question or offer breaks (legit 95% → 80%), and KL is 15–20× the gate's. IFEval goes *up* 4 points, likely because shorter answers trip fewer constraints.
   - **Regex output filter:** removes exactly the phrases it lists and nothing else (38% left). It's also a pipeline change, not a drop-in checkpoint.
5. **Residual failures (v3, clean set, 27 detections).**
   - About 10 are detector false positives: requested emails ("let me know if you can bring anything"), an RSVP line, dialogue and list items. That puts the detector's noise floor around 3 points, and the true residual at about 6%.
   - The real survivors are mostly bold-wrapped `**To help me…**` blocks and offers glued onto the last content paragraph ("Does that make sense? Do you want me to give you another example?"). The gate only fires at block starts.
   - Cosmetic cost: about 1.3% of answers end on a dangling `---`, where the gate stopped right after the rule.

## Before / after (clean set, v3)
- *Instagram bio, 4 options.*
  - Base ends: "> Unlocking the secrets of the brain – accessible to all. 🧠 / To help me refine it even further, could you tell me: * **What kind of content do you primarily share?** …"
  - Gate ends: "> Unlocking the secrets of the brain – accessible to all. 🧠"
- *Group-chat text with a brunch pun.*
  - Base ends with "--- To help me tailor it even more to your group chat, could you tell me: * **What's your group chat usually like?** …"
  - Gate ends after the last option.
- *A post of under 175 words on outsourcing.*
  - Base ends: "#smallbusiness #outsourcing … --- Would you like me to tweak this further, perhaps adding a specific example…?"
  - Gate ends at the hashtags.

## Method caveats
- **Hand-built, iterated recipe.** v1 → v2 was diagnosed by reading v1's failures on `heldout`, and v3 dropped the phrase split. That's why `heldout2` exists; quote heldout2 numbers. The threshold (θ = 5.1) was chosen by reading the top-scoring practice positions.
- **No blind agent sessions yet.** Whether an agent finds this mechanism unprompted is untested. The usual protocol (3 blind Opus sessions, budgeted tools ported from `langmix/agent_tools.py`, matched random edits) is the natural next step.
- **Sign-offs are one tell.** This doesn't touch "Okay, here's…" openers, the vocabulary, em dashes or markdown habits.
- **Measurement:** n = 300 per set and one sample per prompt. The detector is regex-based, with roughly 3 points of false positives on requested text. Truncation at 1,024 tokens (8–14%) hides some endings.
- **Small model:** gemma-3-1b's MMLU (37.6) is near chance on some subjects, so the MMLU guard is weak. IFEval (n = 100) has about ±10-point intervals.
