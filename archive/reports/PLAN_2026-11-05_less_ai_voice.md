# Proposal: make a chat model sound less like AI with a weight edit

**Status (2026-11-05):** the model check (§3) is done; see `RESULTS_2026-11-05_voice_check.md`. Three of four candidates pass the kill rule. gemma-3-4b-it is recommended, and Llama-3.2-3B is out. The sign-off warm-up (idea C) has also been run on gemma-3-1b-it as a hand-built recipe. A single repurposed neuron cuts sign-offs from 58% to 9% on a clean held-out set at no measurable cost; see `RESULTS_2026-11-05_signoff_gate.md`. No blind agent sessions and nothing on 4B yet. This is the user's follow-up idea from `RESULTS_2026-11-05_practical_edits.md` ("Follow-up ideas").

**Same rules as the practical-edits run:**
- one model, picked after a short check
- 3 blind agent sessions, each with a matched random edit
- no training after any edit
- cheap baselines as arms
- a ≤30-line recipe script after the sessions

**Citations below were checked against the web on 2026-11-05** (arXiv or official pages), unless they are marked *unverified*.

---

## 1. Why this might work, given what we've already seen

The AI voice isn't one thing. It's a bundle of tells, and they probably sit in different places in the network. Two of our earlier recipes already match parts of the bundle:

| Earlier result | What it showed | AI-voice tell it maps onto |
|---|---|---|
| **Whisper** (exp. 3) | The model *knew* there was no speech. A prior ("Thank you for watching") beat the honest answer. A gated unit in the last MLP that reads a "no speech" direction and writes " ." fixed it. | **Closers.** The model has finished answering, but the prior adds "I hope this helps! Let me know if…". A gate that reads "answer complete" and writes end-of-turn is the same recipe. |
| **Language mixing** (exp. 2) | Leaks were word-level competition at `lm_head`, and a context-gated logit penalty on a family of rows stopped them. | **Lexicon.** "delve", "tapestry", "testament", "intricate", "pivotal" and the rest compete at the output the same way. |
| **RL head start** (FINDINGS.md) | Opus found a real, causal "hedging" direction in Qwen3.5-0.8B. | **Register.** Hedges, "It's important to note", disclaimers. |
| **Over-refusal** (exp. 1) | A behaviour direction overlapped a useful one (harmful refusal), so selectivity was capped. | **The main risk here.** The voice may overlap "being helpful". |

**The new angle: the voice was added by post-training.**
- Base models don't write like this. West & Potts (2025) find base models beat aligned ones at randomness and creative originality.
- Instruction tuning collapses diversity, and Selective Layer Restoration (SLR, arXiv 2602.06665, 2026) partly undoes that. It resets one contiguous block of layers to the base model's weights. The restored layers were 12–17 in Llama-3.1-8B, 12–27 in Qwen-2.5-7B and 17–27 in Gemma-2-9B. On creative writing, diversity rose 42.5% and quality fell 2.1%. **SLR never measured AI tells**, so whether it removes them is an open question, and it gives us a ready-made weight-edit baseline.
- So the instruct-minus-base delta, ΔW, contains the voice *and* the helpfulness. If the voice lives in a few low-rank components of ΔW, we can remove those and keep the rest. That's testable, and nobody seems to have tested it.

**Framing note.** The target is the tells that annoy readers, not AI-detector evasion. Detectors are used only to measure, never to choose or tune an edit. Sun et al. (2025) find model identity survives paraphrase and word shuffling, so we expect detectors to barely move. If that happens, it's a finding in its own right.

---

## 2. Taxonomy of tells, and where each one probably lives

Sources:
- Wikipedia's "Signs of AI writing" (WikiProject AI Cleanup): the best practical catalogue.
- Kobak et al., "Delving into LLM-assisted writing…" (arXiv 2406.07016): excess vocabulary in about 15M PubMed abstracts.
- Antislop, Paech et al. (arXiv 2510.15061, ICLR 2026): some patterns appear more than 1,000× as often in model output as in human text.
- Shaib et al., "Measuring AI 'Slop' in Text" (arXiv 2509.19163): a taxonomy built from expert interviews, plus span annotations.

| Family | Examples | Where it's probably decided | Cheap fix | Edit idea |
|---|---|---|---|---|
| **Openers** | "Great question!", "Certainly!", "Okay, let's break this down", restating the question | first 1–3 response tokens | ban the first token, or prefill | position-gated unit at the response start |
| **Closers** | "I hope this helps!", "Let me know if…", "In summary," recap, offers of further help | end-of-turn token vs closer, after the content | stop strings (fragile) | **Whisper recipe:** an "answer complete" gate that writes end-of-turn |
| **Lexicon** | delve, tapestry, testament, intricate, vibrant, pivotal, showcase, underscore, realm, navigate | `lm_head` competition | logit bias, or the Antislop sampler | **langmix recipe:** a family penalty along a shared "AI register" direction, which should also hit *unlisted* synonyms |
| **Constructions** | "It's not X, it's Y", tricolons, "Here's the thing:", a rhetorical question then its answer | multi-token, mid layers | regex backtracking (Antislop) only | difference-of-means direction, projected out |
| **Formatting** | headers, bullets and bold for a casual question; emoji; em dashes | an early "format mode" decision | system prompt "no markdown", which also breaks lists the user asked for | **conditional:** gate on "the user asked for structure" |
| **Register** | flattery, hedges, "It's important to note", "consult a professional", forced positivity | persona-like directions (Anthropic's persona vectors, arXiv 2507.21509, include sycophancy) | system prompt | projection, or a persona vector baked into a constant neuron, as agents 4 and 5 did in exp. 1 |

**Whack-a-mole control.** Split every family's patterns into **practice** patterns, which the agent's tell counter sees, and **held-out** patterns, which appear only in evaluation. A ban on "delve" that just produces "dive into" should show up as rising held-out tells.

---

## 3. Models

Requirements:
- a strong AI voice
- a released **base** checkpoint, for the ΔW and SLR arms
- ≤ 4B parameters, to fit the RTX 4070 (12 GB) with the agent's activation tools

| Candidate | Cached? | Base available | Why |
|---|---|---|---|
| **google/gemma-3-4b-it** (+ gemma-3-4b-pt) | no, about 8.6 GB | yes | *Unverified, from memory:* Gemma 3 is markdown-, emoji- and "Okay, let's…"-heavy. 8.6 GB in bf16 fits for inference. |
| **meta-llama/Llama-3.2-3B-Instruct** (+ Llama-3.2-3B) | instruct yes | yes | classic "I hope this helps" closers and list-everything formatting |
| Qwen3-1.7B, non-thinking (+ Qwen3-1.7B-Base) | no | yes | cheap; same family as our hedging finding |
| google/gemma-3-1b-it (+ 1b-pt) | instruct yes | yes | the cheapest fallback |

**Check (about 45 min).** Generate on 60 casual WildChat or no_robots prompts with each candidate. Count tells per response with the rule counter (§5). Pick the candidate with the highest tell density that also has a base model. Kill rule: if every candidate averages fewer than 1.5 tells per response, the voice is too weak at this size. In that case, either download a 7–8B model in 4-bit (with lower agent-tool fidelity) or stop.

Phi-3-mini is out: it is very AI-voiced, but no base model was released.

---

## 4. Datasets

**Prompts where the voice shows up:**
- **allenai/WildChat-1M**: real user prompts. Filter to English, single-turn, not code, not toxic. This is the main target suite, because it's what people actually type.
- **Infinity-Chat** (Artificial Hivemind, NeurIPS 2025): 26K open-ended real queries in 6 categories (creative, brainstorm, …), where models are measurably homogeneous.
- **HuggingFaceH4/no_robots** (*cached*): 10k prompts with **human-written** answers, by category (Chat, Open QA, Brainstorm, Generation, Rewrite…). Used as the human reference and for paired contrasts.
- **OpenAssistant/oasst2**: human-written assistant turns. A second human reference.

**Paired data for finding directions (practice only):**
- **Self-paired minimal pairs (recommended).** The model's own answer next to a "de-slopped twin": the same content with the tells removed. Rewrites come from Claude subagents under fixed rules. Content is held constant, so a difference of means isolates *voice* rather than content. Human-vs-model pairs mix the two.
- **pangram EditLens data** (HF; arXiv 2510.03154, ICLR 2026): human text plus AI-edited versions of the same text, so the pairs run in the opposite direction. 60k training examples, CC BY-NC-SA.
- **HumanLLMs/Human-Like-DPO-Dataset** (arXiv 2501.05032): 10.9k items of (question, human-like answer, formal answer). Caveat: the "human-like" side was generated by Llama-3 70B/405B, so it's a register contrast, not real human text.
- **HC3** (human vs ChatGPT-3.5 answers): dated voice, so use it at most as a sanity check.

**Pattern lists:**
- the Kobak excess-word list
- Paech's `slop-forensics` lists (over-represented words, bigrams and phrases) and `auto-antislop` (MIT)
- regexes taken from Wikipedia's "Signs of AI writing"
- *re-profiled on the chosen model* against no_robots human answers, as slop-forensics does, because tells differ between models

---

## 5. Measurement

| Metric | How | Role |
|---|---|---|
| **Tell rate** | Rule counter: opener and closer regexes; lexicon hits per 1k words; em dashes per 1k words; markdown density (headers, bullets, bold per 100 words) on prompts that don't ask for structure; "not X, it's Y" and disclaimer regexes. Reported per family, and as **practice vs held-out** patterns. | primary, automatic |
| **Blind pairwise "which did a person write?"** | Claude subagents judge exported A/B pairs, as in the refusal spot-check, with the order randomised. Plus **the user blind-reads 20 pairs**, which is also a blog quiz. | primary, perceptual |
| **Helpfulness** | blind pairwise "which answer is more useful?" | guard |
| **Format-request compliance** | about 100 prompts that explicitly ask for a list, table, formal letter, emoji or headings. Does the model still comply? | **conditional guard**, like "Chinese prompts still get Chinese" in exp. 2 |
| IFEval-100, MMLU-500, KL on chat | existing code (`langmix/ifeval_check.py`, refusal MMLU) | guards |
| Length ratio vs base | words per answer | reported, so "say less" doesn't pass as "sound human" |
| **Open Pangram** `editlens_roberta-large` (355M, CC BY-NC-SA, open weights) | mean AI score | secondary. *Prediction: moves little.* Never shown to the agent. |

Judge bias: an LLM judge may prefer AI-voiced text. The "which did a *person* write" framing plus the user's blind read is the check on that.

---

## 6. Ideas, ranked

### A. "Un-slop" edit, the main experiment (recommended)
One model, three blind sessions. The agent's target is the aggregate tell rate across openers, closers, lexicon and formatting, the four families with clean metrics. Constructions and register are measured but not targeted.

**Agent gets:**
- practice prompts plus the model's practice answers
- self-paired minimal pairs
- a tell counter for the **practice** patterns only
- the langmix-style tools (`generate`, `teacher_forced` with activations, `save_edit`), with `lm_head` and the norms in scope
- **read access to the base checkpoint's weights and activations.** This is new; practitioners have the base model.
- budget: ≤ 1% of entries, 30 passes, about 40 min

**Arms:**

| | Arm | Note |
|---|---|---|
| 1 | base | |
| 2 | style system prompt (built from Wikipedia's "Signs of AI writing": no openers or sign-offs, prose unless asked, plain words) | the cheapest fix. Prediction: strong on format and openers, hurts IFEval and format-request compliance |
| 3 | ban list: logit bias on the practice lexicon plus first-token openers (Antislop-style) | prediction: listed words go to ~0, held-out synonyms rise, no effect on formatting. Not available in ollama |
| 4 | **SLR**: restore a contiguous layer block to base, chosen as in the paper by grid search on a proxy | **the literature's weight edit** |
| 5 | **ΔW interpolation** W_inst − λ·ΔW, λ swept | dose curve, for matched comparisons as in exp. 1 |
| 6 | difference-of-means "voice direction" from the minimal pairs, projected out of late writes | abliteration analogue, the strongest script arm |
| 7 | agent ×3 | blind |
| 8 | matched random edit ×3 | |
| 9 | recipe script | after the sessions |
| *10* | *FTPO via auto-antislop, or DPO on Human-Like-DPO* | *only if an agent passes* |

**Pass bar (per session; a draft to argue with):**
- total tell rate on held-out WildChat and Infinity-Chat ≤ ½ of base (strict: ≤ ⅓)
- **held-out pattern tells don't rise** (no whack-a-mole)
- the "person wrote it" judge prefers the edit over base on ≥ 65% of pairs
- helpfulness win-or-tie vs base ≥ 45%
- format-request compliance within 5 points of base
- IFEval and MMLU within 1 point (significance rule from exp. 2)
- length within ±25% of base
- random control fails

**The experiment succeeds** if 2 of 3 sessions pass **and** the agents beat SLR and ΔW interpolation (arms 4–5) on tell rate at matched helpfulness.

**Predictions:**
- Openers and closers fall to near 0, since the Whisper-style gate fits them.
- Lexicon falls a lot, and partly on held-out words if the agent edits a shared direction rather than rows.
- Formatting is the hardest *conditional* part.
- The detector barely moves.
- SLR reduces tells but costs format-request compliance and IFEval, because it restores base behaviour wholesale.

**Time:** about 1.5 days.
- Build 4–5 h: suites, tell counter, pair generation, judge export, porting the langmix tools, SLR and ΔW scripts.
- Sessions: 3 × 40 min.
- Evaluations 3–4 h: 300–500-token answers dominate.
- Judging: about 1 h.

### B. "Where is the slop in ΔW?" (most novel; can run inside A as an analysis)
Per matrix, take the SVD of ΔW = W_inst − W_base. Score each singular component by how much its removal lowers the tell rate per unit of IFEval lost, using the voice direction from arm 6 as a cheap first filter. Then remove the top-k components.

The question is whether the voice is **low-rank and separable** in the post-training delta. If it is, it gives a "de-voiced instruct model" built from nothing but base and instruct weights, and it ties into the chat-vector and model-merging literature. As a scripted arm, add about 0.5 day.

### C. The sign-off gate (smallest, highest odds, a good warm-up)
Closers only. Use the Whisper recipe directly:
1. Find a direction that separates "content finished, closer next" from "content continues", at the position before a closer.
2. Repurpose the least-used last-layer neuron: fc1 = that direction, bias = midpoint threshold, fc2 = raise end-of-turn.

Baselines: stop strings, and a closer ban. Pass if closers fall ≥ 80%, truncated answers don't rise, and the guards hold. **About half a day.** If it works, it's a clean blog beat: the same 29-line recipe, now in an LLM.

### D. Lexical family generalization (langmix recipe)
Penalise a "fancy register" direction in unembedding space rather than single rows. The test: does an edit fit on 30 practice slop words suppress **30 held-out** ones that a ban list can't touch? This is a direct head-to-head with the Antislop sampler (8,000+ patterns at inference) on the one axis an edit could win: no sampler, and generalization to unlisted words. About 1 day, or run as a sub-target of A.

### E. Conditional formatting
Prose for casual questions, structure when asked. This is the purest version of the project's "conditional edit" bet, because the system-prompt baseline fails it by design. It's the hardest one, so it belongs in A's measurement rather than as a standalone run.

---

## 7. Risks

- **The voice overlaps helpfulness**, as in exp. 1. Removing it may give terse, base-model-ish answers. The helpfulness, length and format-compliance guards catch this.
- **Small models may not be very sloppy.** The check catches this.
- **Judge self-preference.** Use the "person wrote it" framing and the user's blind read.
- **Goodhart on the counter.** Use held-out patterns, and keep the detector and judge hidden from the agent.
- **Licensing.** EditLens data and Open Pangram are non-commercial (CC BY-NC-SA). They're fine for a research blog; keep them out of anything product-facing.

## Recommendation
1. Run the model check (about 45 min).
2. Then **C as a warm-up** (half a day). It reuses the Whisper recipe and gives an early read.
3. Then **A, with B as scripted arms**. Fold D and E in as measured sub-targets rather than separate experiments.

## Sources
- Antislop, Paech et al. 2025: https://arxiv.org/abs/2510.15061 (code: github.com/sam-paech/auto-antislop)
- Measuring AI "Slop" in Text, Shaib et al. 2025: https://arxiv.org/abs/2509.19163
- Kobak et al., excess vocabulary: https://arxiv.org/abs/2406.07016
- Idiosyncrasies in LLMs, Sun et al. (ICML 2025): https://arxiv.org/abs/2502.12150
- Artificial Hivemind / Infinity-Chat (NeurIPS 2025): https://neurips.cc/virtual/2025/poster/121421
- Base Models Beat Aligned Models at Randomness and Creativity, West & Potts: https://arxiv.org/abs/2505.00047
- Selective Layer Restoration: https://arxiv.org/abs/2602.06665
- Enhancing Human-Like Responses / Human-Like-DPO-Dataset: https://arxiv.org/abs/2501.05032
- EditLens and Open Pangram: https://arxiv.org/abs/2510.03154, https://www.pangram.com/blog/introducing-open-pangram
- Persona vectors: https://arxiv.org/abs/2507.21509
- Binoculars (heavier zero-shot detector; not planned): https://arxiv.org/abs/2401.12070
- Wikipedia: Signs of AI writing: https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
