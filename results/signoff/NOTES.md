# signoff: gemma-3-1b-it's "Would you like me to…" endings

Task code: `tasks/signoff.py` (+ `tasks/_signoff_detect.py` detector, `tasks/signoff_sweep.py` sweep variant,
`tasks/_signoff_summary.py` tables/examples). Port of the original sign-off gate experiment
(`RESULTS_2026-11-05_signoff_gate.md`, v3 recipe). Everything below is greedy decoding on the clean held-out set.

## The failure
Gemma-3-1b-it ends most answers by talking about the conversation instead of stopping: "Would you like me to tweak
this further?", "Let me know if…", "To help me tailor this even more, could you tell me: * …". Base rate on the
300 held-out prompts: **59.0%** (original run, sampled: 58.3%). Human no_robots answers: 0%.

## Design
- **Items** (identical construction to the original, verified id-for-id): no_robots *train*, single-turn, not
  Coding/Chat, < 2,500 chars, shuffled with `random.Random(0)`. Practice = first 300; held-out = next 300 (the
  original "heldout2", never inspected) + the 40 hand-written **legit** prompts (requested text should itself contain
  a question or an offer; `legit=True`, scored separately).
- **Practice labels** come from the base model's own greedy practice replies: positive = reply has a sign-off;
  replies that hit `max_new` (34/300) are ignored. 185 positive, 81 negative.
- **Probe sites** (`probe_sites` override, v3): positive = the position before each sign-off block start (any
  phrasing; plus every non-list block start inside the sign-off, e.g. the anchor after a `---`), 271 sites.
  Decision negatives = the position before every *content* block start (2,112; the hardest negatives, used for the
  `signal` AUC). Content = every position that predicts a content token (115,718). Content block starts are in both
  sets, so the threshold sits above them too.
- **Gate**: last layer (25), target `<end_of_turn>`; the framework picks the suppressed tokens from the base's next
  tokens at positive sites: `---`, `**`, `To`, `Do`, `Would`, `Let`, `I`, `*`. Unit 5006, θ = 5.66 (framework rule:
  max content score + 0.1), 3,455 entries changed. Practice: 40% of positive *sites* above θ, 0 content positions.
- **LoRA targets** (`sft_target`): positives → base reply cut right before the sign-off block (then end of turn);
  negatives → base reply unchanged; truncated replies skipped. 266 examples, framework recipe (r=16, 3 epochs, 200 steps).
- **Prompt fix**: the original run's system prompt, verbatim: "Answer the request and stop when the answer is
  complete. Do not end with offers of further help, follow-up questions to the user, or suggestions of what else
  you could do."
- **Signal**: cross-validated probe AUC (sign-off start vs content block start) 0.991–0.994 at layers 17–25. Go.

### Metrics (`score`)
- `signoff` (headline failure): % of the 300 held-out replies with a sign-off (old detector, unchanged regexes).
- `content_kept` (headline kept): the original "content kept vs base" — total characters before any sign-off, divided
  by the same for the base reply (stored at `data` time as `ref`; the base arm reproduces it exactly), x100.
- `content_identical`: % of replies whose content (text before any sign-off) equals base's character for character.
- `legit_kept`: % of the 40 legit prompts whose reply still contains the requested question (`?`) / offer (old rule).
- `appendix` (meta sections such as "**Important Note:**" late in the content), `dangling_rule` (reply ends on `---`),
  `truncated` (hit `max_new`), phrase split (`signoff_practice_phrase` / `signoff_heldout_phrase_only`, the
  original split of anchor phrases; the v3 gate and the LoRA are trained on all phrasings, so it is descriptive only).

## Deviations from the original run
1. **Greedy decoding** (framework) instead of Gemma's shipped sampling, one reply per prompt.
2. **max_new = 1,024, not 768.** Base held-out replies at 768 tokens would be truncated 31% of the time (94/300,
   measured on the stored references), hiding the very endings we score; at 1,024: 13%. Same budget as the original.
3. **Threshold rule**: the original v3 θ (5.1) was hand-picked by reading top-scoring practice positions; here the
   framework rule (θ = max content score + 0.1 = 5.66) is used unchanged. Direction and unit also come from the
   framework fit (logistic probe on sign-off sites vs content + content-block-start sites; least-used unit 5006 vs 3412).
   This is the main reason the gate is weaker than the original v3 (see Results).
4. **No calibration set / replay (`n_calib = 0`).** The framework's label-free calibration doubles as LoRA replay,
   whose targets are 128-token greedy replies cut mid-answer with `<end_of_turn>` appended. For a task whose
   failure *is* where the turn ends, that would teach the LoRA to stop early. The original run had no calibration
   set either. (I did not get to measure how many calibration replies contain sign-offs; a CPU check was cut to free the
   machine for the sweep.)
5. **Curly quotes**: the detector now normalises ’ to ' before matching (framework-wide fix). On the base references
   this changes 0 labels and 1 sign-off start position; practice labels are unchanged, so the gate/LoRA fits are unaffected.
6. **Guards**: MMLU-200 (framework) instead of MMLU-500; chat KL on 60 greedy chat replies (framework); no IFEval.
7. **Output files**: the framework stores outputs cut at 1,500 chars, but the sign-off is at the end of replies that
   average 2,200 chars. `score()` therefore also writes full texts to `runs/signoff/fulltexts/<hash>.jsonl` and
   transparently uses them when `patchkit.run rescore` passes it the cut outputs (verified: rescore reproduces
   every metric). The full replies for any arm are in that folder.

## Results (300 clean held-out prompts + 40 legit; greedy, max_new 1,024)

| Arm | Size | Sign-off % | Content kept vs base | Content identical % | Legit kept % | Appendix % | Dangling `---` % | Truncated % | MMLU-200 | Chat KL |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 59.0 | 100.0 | 100.0 | 97.5 | 26.0 | 0.0 | 13.0 | 41.0 | 0 |
| prompt (system message) | 0 | 29.0 | **58.9** | 0.0 | 87.5 | 5.3 | 0.0 | 0.3 | – | – |
| **gate** (1 neuron) | 3,455 entries | 23.3 | **99.6** | 69.3 | 95.0 | 25.7 | 2.3 | 11.0 | 41.0 | 0.018 |
| random (same unit, same norms) | 3,455 entries | 62.0 | 103.2 | 2.0 | 97.5 | 30.7 | 0.0 | 20.0 | 42.5 | 0.148 |
| **LoRA** r=16 | 13.0M params | **8.3** | 95.9 | 2.3 | 95.0 | 25.3 | 0.3 | 7.0 | 41.5 | 0.098 |

95% CI on a rate near 59% with n=300: about ±5.6 points; near 10–25%: ±3.5–5 points. Legit (n=40): ±7–10 points.

Phrase split (practice-phrase / held-out-phrase-only sign-offs): base 19.0 / 40.0, prompt 17.3 / 11.7,
gate 7.0 / 16.3, random 22.3 / 39.7, LoRA 2.3 / 6.0. Of the held-out prompts where base signs off, 39% still sign off
under the gate, 12% under LoRA, 41% under the prompt fix.

**Reading it**
- **LoRA wins on the failure rate** (59 → 8.3%), the gate cuts it by 60% (→ 23.3%), the system prompt by half (→ 29%).
- **The gate is the only fix that leaves the answer alone**: 99.6% of content characters kept, and 69% of replies
  have byte-identical content to base (44% are identical as whole replies). LoRA keeps 96% of the length but rewrites
  nearly everything (2.3% identical content) — it is a different model that writes similar answers. The prompt fix
  removes 41% of the content (answers shrink from 2,185 to 1,256 chars) and breaks 1 in 10 requested questions/offers.
- Where the gate's content is not identical, the divergence mostly starts early (median ~half-way into the content), so it
  is greedy decoding flipping on near-ties from a tiny mid-answer perturbation, not truncation; the random edit, with a
  ~8x larger KL, diverges on 98% of replies.
- **Cost of the gate**: 2.3% of replies end on a dangling `---` (the gate stopped right after the rule that precedes
  a sign-off; original run: 1.3%). The random control confirms the effect is not the perturbation (62% sign-offs, KL 8x).
- **vs the original v3** (sampled, hand-picked θ = 5.1): 58.3 → 9.0% sign-offs at 97.6% content. The framework's
  conservative threshold (above every one of 115k practice content positions) only lets the gate fire at 40% of
  practice sign-off sites. Lowering θ toward the original would be tuning; it is not done here.
- MMLU-200 is flat everywhere (41.0–42.5; near chance for a 1B model, so a weak guard).

## Data-scaling sweep (gate vs LoRA)

Task `signoff_sweep` (same practice data, base replies, sites, targets and scoring) on a **reduced held-out set:
the first 64 of the 300 held-out prompts, no legit prompts** (GPU time on a shared card; fixed before any sweep
result). On these 64: base 62.5% sign-offs, prompt fix 31.2%, full gate 21.9%, full LoRA 10.9% (main-run outputs
rescored on the subset). n = labelled practice items, stratified (≈70% positives); "300" = all practice items
(266 labelled). Gate on all data is deterministic (1 run); everything else 3 seeds. Mean ± sd over seeds.

| Method | n practice | Runs | Sign-off % (base 62.5) | Content kept % | Content identical % | Chat KL | Gate: practice sites above θ |
|---|---|---|---|---|---|---|---|
| gate | 16 | 3 | **5.2 ± 2.4** | 97.1 ± 1.1 | 56.2 ± 5.6 | 0.042 | 100% |
| gate | 32 | 3 | 9.9 ± 2.4 | 97.4 ± 1.8 | 64.6 ± 5.9 | 0.031 | 100% |
| gate | 64 | 3 | 10.4 ± 4.5 | 99.3 ± 0.3 | 67.2 ± 0.0 | 0.024 | 100% |
| gate | 128 | 3 | 22.9 ± 13.7 | 99.8 ± 0.7 | 68.8 ± 1.6 | 0.019 | 24–58% |
| gate | 300 | 1 | 21.9 | 100.5 | 70.3 | 0.018 | 40% |
| LoRA | 16 | 3 | 26.6 ± 8.7 | 98.3 ± 2.9 | 3.6 ± 0.9 | 0.117 | – |
| LoRA | 32 | 3 | 37.0 ± 5.1 | 103.1 ± 3.1 | 3.6 ± 0.9 | 0.075 | – |
| LoRA | 64 | 3 | 22.9 ± 4.5 | 99.1 ± 3.6 | 4.7 ± 0.0 | 0.072 | – |
| LoRA | 128 | 3 | 14.1 ± 1.6 | 98.3 ± 1.5 | 4.2 ± 0.9 | 0.081 | – |
| LoRA | 300 | 3 | 12.0 ± 2.4 | 96.8 ± 0.3 | 4.2 ± 0.9 | 0.098 | – |

95% CI for a single run on 64 prompts: ±6–12 points. Treat differences under ~10 points as noise.

**Reading it**
- **The gate gets *worse* with more data**, and the reason is mechanical: the framework sets θ just above the
  highest-scoring of all practice content positions. With 16 items that is ~6k positions and the gate fires at 100%
  of practice sign-off sites; with 128+ items (55k–116k positions) the maximum is set by a few outlier content
  positions (in the original run the top-scoring content positions were mostly starts of meta appendices such as
  "**Notes on the Poem:**"; not re-checked here), and only 24–58% of sign-off sites clear it. Extreme-value thresholds grow with sample size.
- **With ≤ 64 practice items the gate beats LoRA by a wide margin** (5–10% vs 23–37% sign-offs) while keeping
  97–99% of content at ~3x lower chat KL. LoRA needs ~128+ items to reach 12–14%, and with the full set it is
  better than the full-data gate (12.0 vs 21.9 here; 8.3 vs 23.3 on all 300 prompts).
- Small-n gates trim a bit more content (97% kept at n=16 vs 99–100% at n≥64): a lower θ also fires on some
  appendix starts. LoRA at every n rewrites ~96% of answers (content identical 3–5%).
- Suggestion for the core (not done here; no tuning on held-out): a threshold set from a high quantile of content
  scores, or from a fixed-size content sample, instead of the max over all positions, would make the gate's
  operating point independent of n. This sweep suggests it would also make the full-data gate much stronger.

## Before / after examples (held-out, greedy; last ~300 characters)

### Three haiku about overthinking in the shower (held-out id 338)

*Gate: the base answer byte for byte, minus the rule and the sign-off. LoRA: no sign-off, but different haiku.*

**Prompt:** I want a series of 3 haiku that depict overthinking in the shower till the water cools off and your fingers are pruny.

**Base** (619 chars) ends:

```
…ght, frozen grip,
Coldness chills the skin.

**Haiku 3**

Water fades to gray,
Prickling touch, a quiet dread,
Lost in what could be. 

---

I hope these capture the feeling you were going for!  Let me know if you'd like me to try a different approach or focus on a specific aspect of the experience.
```

**Gate** (452 chars) ends:

```
…ers:

**Haiku 1**

Warm water descends slow,
Pruny fingers, thoughts take hold,
Stillness, fear takes root.

**Haiku 2**

Steam obscures the view,
Thoughts swirl, a tight, frozen grip,
Coldness chills the skin.

**Haiku 3**

Water fades to gray,
Prickling touch, a quiet dread,
Lost in what could be.
```

**LoRA** (441 chars) ends:

```
…s:

**Haiku 1:**

Cold water descends slow,
Pruny fingers, thoughts take hold,
Stillness, fear takes root.

**Haiku 2:**

Steam obscures the mind,
Thoughts like droplets, cling and fall,
Quiet, then the chill.

**Haiku 3:**

Water softly sighs,
Pruney skin, a quiet ache,
Lost in thought's deep pool.
```


### A rhyming poem about light and dark (held-out id 499)

*Gate: identical poem, stops after the last couplet. LoRA: a different, shorter poem.*

**Prompt:** Write a poem that rhymes, about light and dark, night vs day.

**Base** (962 chars) ends:

```
…ft and low.

And in this balance, we can see,
The beauty of tranquility,
A timeless dance, forever bright,
Of darkness and the golden light. 

---

Would you like me to:

*   Try a different style (e.g., haiku, free verse)?
*   Focus on a particular aspect of light and dark (e.g., emotions, nature)?
```

**Gate** (802 chars) ends:

```
…silver shimmer, soft and slow.

For night and day, a constant plea,
A different kind of majesty,
One brings the warmth, the vibrant glow,
The other whispers, soft and low.

And in this balance, we can see,
The beauty of tranquility,
A timeless dance, forever bright,
Of darkness and the golden light.
```

**LoRA** (556 chars) ends:

```
…rld, a gentle stroke.

The stars ignite, a diamond dust,
In darkness born, a sacred trust.
The moon ascends, a silver grace,
Illuminating time and space.

Though dark and still, a peaceful might,
The light returns with warming light.
For day and night, a constant flow,
A beautiful and wondrous show.
```


### Three sentences for an "about me" page (held-out id 588)

*The "To help me tailor these…" family. Gate: identical three sentences. LoRA: rewritten sentences.*

**Prompt:** Write 3 sentences for an about me section on a website. It should include how I am a great leader and that I graduated summa cum laude.

**Base** (914 chars) ends:

```
…set to any team and a passionate advocate for growth.

To help me tailor these even further, could you tell me:

*   **What kind of website is this?** (e.g., a freelance website, a professional profile, a personal blog?)
*   **What is your field/industry?** (e.g., marketing, engineering, education?)
```

**Gate** (667 chars) ends:

```
…nging a strong foundation of knowledge and a proven ability to analyze complex challenges and develop innovative solutions.
3.  My dedication to excellence, coupled with my leadership skills honed during my academic journey, makes me a valuable asset to any team and a passionate advocate for growth.
```

**LoRA** (710 chars) ends:

```
…d] and a dedication to strategic thinking, I bring a unique blend of analytical skills and visionary leadership to every project.”
3.  “I’m genuinely excited to share my experience and commitment to excellence, consistently striving to inspire and motivate others while delivering impactful results.”
```


### An intro paragraph for an article on the Manhattan Project (held-out id 573)

*Gate: identical paragraph, ends at the closing quote.*

**Prompt:** Please write a paragraph for an article about the Manhattan Project. The goal of this paragraph should be to introduce the rest of the article that would be written afterward.

**Base** (1214 chars) ends:

```
…elp me tailor this even further, could you tell me:**

*   **What is the overall tone of the article?** (e.g., scholarly, narrative, analytical?)
*   **What specific aspects of the Manhattan Project are you focusing on?** (e.g., the scientific process, the political context, the moral implications?)
```

**Gate** (901 chars) ends:

```
…to Oak Ridge in Tennessee. This article will explore the key scientific breakthroughs, the ethical dilemmas faced by the project’s leaders, and the lasting impact of this extraordinary effort on the 20th century and beyond, ultimately examining the profound and controversial legacy of its creation.”
```

**LoRA** (861 chars) ends:

```
…develop the first atomic bombs.  This article will explore the scientific breakthroughs that fueled the project, the ethical dilemmas it presented, and the profound impact it had on the course of the war, ultimately leaving a lasting legacy of both technological innovation and profound controversy.”
```


### A chore list ("Do not do any additional variations.") (held-out id 463)

*Imperfect: the gate stops the offer, but the model then writes an "Important Note" appendix instead (the gate fires only where the sign-off would start; the next block can be a different tell).*

**Prompt:** Make a to-do list of common chores that need to be completed to keep a clean, healthy home. The chores may be listed by required frequency or as tasks to complete throughout the week as a whole. Do not do any additional variations.

**Base** (1152 chars) ends:

```
…  Clean gutters (if applicable)

**Other (As needed)**

*   Empty all trash cans
*   Clean pet areas (litter boxes, etc.)
*   Wipe down door handles and light switches

---

Would you like me to:

*   Categorize the list further (e.g., by room)?
*   Add specific instructions for any of these chores?
```

**Gate** (1164 chars) ends:

```
…ers (if applicable)

**Other (As needed)**

*   Empty all trash cans
*   Clean pet areas (litter boxes, etc.)
*   Wipe down door handles and light switches

---

**Important Note:** This is a general list. Adjust the frequency and tasks based on your home’s size, lifestyle, and personal preferences.
```

**LoRA** (901 chars) ends:

```
…efrigerator)
*   Clean windows and mirrors
*   Wipe down baseboards
*   Clean light fixtures
*   Declutter a specific area (e.g., a drawer, shelf)

**Seasonal (Every 3-6 months)**

*   Deep clean carpets/rugs
*   Wash curtains/drapes
*   Clean out pantry
*   Clean out closets
*   Wash/replace towels
```


### A travel ad under 200 words (held-out id 555)

*Imperfect: the gate fires after the `---` and leaves a dangling rule (2.3% of gate replies).*

**Prompt:** Write a travel ad for a trip to the Adirondack mountains. Focus on activities there, the scenery, and keep it concise and under 200 words.

**Base** (1131 chars) ends:

```
…y of birds.
* **Scenic Drives:** Explore charming towns and breathtaking vistas.

**Book your Adirondack adventure today!** [Website/Phone Number] #Adirondacks #GetOutside #NatureEscape 

---

Would you like me to tweak this ad in any way (e.g., add a specific activity, target a different audience)?
```

**Gate** (1022 chars) ends:

```
… fishing – the water’s calling!
* **Wildlife Viewing:** Keep an eye out for moose, bears, and a dazzling array of birds.
* **Scenic Drives:** Explore charming towns and breathtaking vistas.

**Book your Adirondack adventure today!** [Website/Phone Number] #Adirondacks #GetOutside #NatureEscape 

---
```

**LoRA** (994 chars) ends:

```
…se, bears, and a dazzling array of birds.
*   **Stargazing:** Escape light pollution and witness a truly spectacular night sky.

**Accommodation:** Cozy cabins, charming lodges, and rustic campgrounds.

**Book your Adirondack escape today!** [Website/Phone Number] #Adirondacks #Getaway #NatureEscape
```

## Framework issues hit (not changed in `patchkit/`; for the core owners)
1. **Outputs are saved cut at 1,500 chars** (`eval_arm`). Sign-offs sit at the end of ~2,200-char replies, so
   rescoring from saved outputs is wrong for this task. Worked around in `Signoff.full_texts` (full-text dump keyed
   by a hash of the cut texts; `rescore` then works). A task-level `output_chars` (or no cut) would be cleaner.
2. **LoRA `max_len = 768` tokens (prompt + target)**: 57 of the 185 positive targets (31%) are longer, so the LoRA
   never sees where those replies should end (their "cut before the sign-off" point is past the window). Most
   negatives fit (10/81 cut). LoRA still won on the full set; with a window ≥ prompt + `max_new` it might do better.
3. **Calibration doubles as LoRA replay** and the replay replies are cut at 128 tokens with `<end_of_turn>`
   appended, which teaches early stopping; hence `n_calib = 0` here. Separate switches (or no eot on truncated
   replay replies) would let this task use calibration for the gate only.
4. **θ = max content score + margin** grows with the amount of practice data (see the sweep). A quantile rule would
   make the gate's operating point independent of n.
5. `capture()` computes full-vocabulary logits for 8 sequences of ~1,100 tokens (≈5.4 GB); it logged CUDA OOM
   warnings at 12 GB but recovered. `logits_to_keep=1`/a hook-only forward would avoid it.
6. Generation batches larger than 64 at 1,024 new tokens (a first sweep attempt used 140) ran ~7x slower per
   item on the 12 GB card (11.9 GB in use, likely spilling; a concurrent CPU job of mine also competed), so the
   sweep uses 64-prompt batches. That attempt's single result was discarded.

## Files
- `results/signoff/{base,prompt,gate,random,lora}.json`, `outputs/*.jsonl` (first 1,500 chars per reply),
  `signal.json`, `items.jsonl` (held-out items carry the base reference reply `ref`).
- Full replies: `runs/signoff/fulltexts/*.jsonl`. Edits: `runs/signoff/edits/`. LoRA: `runs/signoff/lora/full`.
- Sweep: `results/signoff_sweep/sweep/*.json` (+ `outputs/`), full replies in `runs/signoff_sweep/fulltexts/`.
- Logs: `runs/signoff_*.log`. The prompt arm was evaluated twice (the second run, after the curly-quote fix, is the
  one reported; 29.0% vs 28.7% in the first).
