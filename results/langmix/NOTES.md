# langmix: Chinese words leaking into German, Russian and Thai answers

Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B. Task code: `tasks/langmix.py`, with helpers in `tasks/_langmix_text.py`
(scoring, CJK tokens, ban processor) and `tasks/_langmix_recipe.py` (the extra "recipe" arm). It's a port of the old
`langmix/` experiment ("Experiment 2" in RESULTS_2026-11-05_practical_edits.md). The new code doesn't import the old
project.

## Results

**Held-out results.**
- Items: MGSM de/ru/th 120 each, en/zh 60 each.
- Decoding: greedy, 1,024 new tokens.
- Files: `results/langmix/<arm>.json`, with the first 1,500 characters of each output in `outputs/`.
- "recipe" is the extra arm `gate_recipe` (see below).

| Arm | Leak, pooled | de / ru / th | zh reply | Acc de+ru | Acc th | Acc en | Acc zh | Finished de/ru/th | MMLU-200 | Chat KL | Entries changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | **14.7** | 9.2 / 18.3 / 16.7 | **75.0** | 35.0 | 10.0 | 78.3 | 60.0 | 77.8 | 41.5 | 0 | 0 |
| prompt (instruction) | 15.0 | 2.5 / 31.7 / 10.8 | 95.0 | 37.1 | 13.3 | 91.7 | 70.0 | 80.6 | n/a | n/a | 0 |
| **gate** (one neuron) | **23.3** | 15.8 / 49.2 / 5.0 | **18.3** | 2.1 | 0.0 | 55.0 | 13.3 | 12.2 | 44.0 | 0.557 | 4,607 |
| random (same neuron, random) | 26.7 | 10.0 / 4.2 / 65.8 | 45.0 | 18.8 | 4.2 | 53.3 | 30.0 | 68.3 | 41.0 | 0.534 | 4,607 |
| **lora** (r=16, 3 epochs) | **9.2** | 6.7 / 14.2 / 6.7 | **75.0** | 39.6 | 4.2 | 86.7 | 60.0 | 72.8 | 39.5 | 0.744 | 18.5M trainable |
| **recipe** (agents' lm_head penalty) | **0.3** | 0.0 / 0.0 / 0.8 | **68.3** | 35.4 | 9.2 | 78.3 | 53.3 | 77.2 | 41.5 | **0.0018** | 8,135,168 |

**Degenerate repetition loops.** Share of replies containing a short string repeated 9 or more times. Computed from
the full texts in `runs/langmix/full_outputs/`.

| Arm | de / ru / th / en / zh |
|---|---|
| base | 1 / 2 / 24 / 0 / 0 |
| gate | 47 / 59 / 94 / 13 / 62 |
| random | 23 / 28 / 38 / 27 / 27 |
| lora | 2 / 1 / **31** / 0 / 0 |
| recipe | 1 / 2 / 26 / 0 / 0 |

**Compared with the old run** (sampled, 2,048 tokens; `RESULTS_2026-11-05_practical_edits.md`).

| | Old | This port |
|---|---|---|
| Base leak, pooled | 31.4 | 14.7 |
| Leak de / ru / th | 14.2 / 39.2 / 40.8 | 9.2 / 18.3 / 16.7 |
| Instruction prompt: leak | 26.9 | 15.0 (no help) |
| Instruction prompt: zh reply | 85 → 95 | 75 → 95 |
| Agents / recipe: leak | 0.8–1.4 (recipe script 1.7) | 0.3 |
| Agents / recipe: zh reply | 85 → 81–82 | 75 → 68.3 |

- **Greedy decoding halves the base leak rate.** Absolute numbers aren't comparable with the old run; the arm-vs-base
  deltas are.
- **The CJK token ban** wasn't re-run as an arm. On practice, the ban replies leak 3.4% by `mix_flag`: 2 of 59, via
  non-Han script runs. They're used only as LoRA targets.

**Stop-or-go.**
- **Signal.** Cross-validated probe AUC, leak onset vs sampled ordinary positions:

  | Layer | AUC |
  |---|---|
  | 14 | 0.961 |
  | 22 | 0.983 |
  | 27 | 0.982 |

  That's a borderline "go": below the ~0.99 that every working gate had.
- **Fit.** The gate is on layer 27, unit 6831, with θ = 1.27.
  - Recall on practice onsets: **31%**.
  - Content and calibration fire rate on practice: 0.
  - Suppressed tokens: 他已经 金 毫升 第 计算 美元 剩余 约翰.
  - Raised token: " ".

**Verdict: the generic one-neuron gate fails on this task, and it fails by breaking the model, not just by missing.**
- **It doesn't fail because it fires too often.** At threshold it is silent on every practice and calibration
  position.
- **It fails because its off state isn't zero.**
  - The neuron computes `silu(g·(d·x − θ)) · k·(m·x)`, and m·x ≈ 1.
  - At ordinary positions d·x sits only about 1–2 units below θ. So silu sits on its negative tail (about −0.03 to
    −0.27). Multiplied by gain × k = 800, every token gets a constant write of −20 to −65 along v.
  - Since v = (raise " ") − (lower CJK), the off state does the opposite of what's intended. It lowers spaces
    ("dieAufgabenstellung", "durchAddition") and raises Chinese.
  - Measured on practice by `tasks/_langmix_gate_diag.py` (`gate_diag.json`):

    | Practice replies | Mean write along v |
    |---|---|
    | zh | −43 |
    | en | −21 |
    | clean de/ru/th | −32 |
    | leaking de/ru/th | −65 |

    The activation is negative at 100% of positions.
- **Result:** loops ("$maximal$maximal…"), 12% of de/ru/th replies finished, and ru leaks going *up* to 49%.
- **The random control**, with the same neuron and norms, damages the model about as much: KL 0.53 vs 0.56. So
  this is the construction, not the direction.
- **Nothing on held-out was tuned to try to rescue it.** The fix belongs in the core (see "Proposed core changes").

**LoRA** (18.5M trainable parameters, 611 s to train) cuts leaking by about 40% relative: 14.7 → 9.2.
- **Kept:** zh reply (75.0) and de+ru accuracy (35.0 → 39.6).
- **Costs:**
  - Thai accuracy 10.0 → 4.2, and Thai loops 24 → 31% ("ให้ให้ให้…").
  - Finished de/ru/th 77.8 → 72.8.
  - The largest chat KL of any arm (0.74).
- The KL may partly come from the replay pairs, which are truncated 128-token R1 replies followed by end-of-turn. It
  isn't separated here.

**The recipe** (a continuous penalty, not a thresholded neuron, 49 s to fit) removes the leak (0.3%). It leaves
de+ru, en and Thai accuracy, finishing, MMLU and chat KL (0.002) where they were.
- **zh-reply cost:** 75.0 → 68.3, i.e. 4 of 60 items, and zh accuracy 60 → 53.3. The old run saw 85 → 81–82.
- **Breakdown:** zh_answered 75 → 70 and zh_reply_of_answered 100 → 97.6. So most of the drop is Chinese replies
  that no longer finish within 1,024 tokens.
- **What replaces the Chinese:** where a leak would have been, it often writes an English word ("500 dollars",
  "250 dollars в час"). `mix_flag` doesn't count that, and it's arguably still a mixing artifact. The model already
  mixes English into ru/th reasoning at baseline.
- **Edit size:** 8.1M lm_head entries, versus 4,607 for the neuron.

**Sweep: not run.** One LoRA fit takes about 10 minutes and one held-out eval about 13 minutes on the L4, so
`--ns 16,32,64,all --seeds 0,1 --methods gate,lora` would take about 5 hours. With the gate broken by construction,
a gate-vs-LoRA scaling curve wouldn't be informative until the core fix lands.

## Proposed core changes
1. **Gate off state.**
   - Make the neuron exactly (or negligibly) zero at ordinary positions. For example, pick g so that
     g·(θ − median content score) ≥ 12, which puts silu's tail below 1e-4.
   - Or report `off_write = gain·k·mean|silu(...)|` over content sites in `fit_gate`'s meta, and refuse to fit when
     it's large.
   - The entity gate presumably had a large gap. Here the gap is about 1–2 units, at the last layer.
2. **`llm.eot_id`:** fall back to `tok.eos_token_id`. It raises StopIteration for R1-distill tokenizers today; the
   task shims it.
3. **Saved outputs:** keep the full text, or add a task-level `output_chars`. 1,500 characters truncates R1 replies,
   so `rescore` is wrong for this task.
4. **Optional:** a `Task.gen_kwargs` hook (sampling and processors). It's not needed now that greedy works.

## Design

**Data.**
- **Held-out:** MGSM test (juletxara/mgsm). It uses the same problems as the old run (seed-0 shuffle, first n):
  - de / ru / th: 120 each (target; the answer must not leak)
  - en: 60 (must stay English)
  - zh: 60 (must still be answered in Chinese)
- **Practice:** GSM8K **train** problems, so there's no overlap with MGSM (which is built from GSM8K test). The
  translations also come from a different pipeline:
  - de / ru / th: 128 each, machine-translated (Mathoctopus/GSM8KInstruct_Parallel); each problem is used in one
    language only
  - en: 48 (GSM8K)
  - zh: 48 (GSM8K_zh)
- **Prompt:** the question only. No system prompt and no answer-format suffix, the same as the old run's MGSM prompt.
- **Decoding:** greedy (patchkit's default), at most 1,024 new tokens. Metrics are computed on the full generated
  text, including truncated replies.

**Metrics.** These are the old run's definitions, ported as they were.
- **leak_rate** (headline failure): share of de/ru/th replies with at least one Han character anywhere in the
  reasoning or the answer, pooled over the three languages. A run of 3 or more characters of another foreign
  non-Latin script also counts.
- **zh_reply** (headline kept): the final answer (the text after `</think>`) has at least 4 Han characters, and at
  least as many Han characters as Latin words. A reply with no `</think>` counts as False.
- **Also reported:**
  - accuracy per language: the last `\boxed{}` number, else the last number in the answer
  - `acc_deru`
  - `zh_answered` and `zh_reply_of_answered`
  - `en_leak`
  - `answered_deruth`
- **Guards:** patchkit's MMLU-200 and chat KL.

**Labels.**
- `positive = True`: a de/ru/th practice item whose greedy base reply leaks.
- `positive = False`: a de/ru/th practice item that doesn't leak, and every zh/en practice item.
- `data` writes the practice base replies itself, to `runs/langmix/base_practice.jsonl`. It uses `cmd_base`'s
  format, order, batching and decoding. It has to, because the labels, the gate targets and the LoRA targets all
  depend on those replies. `all` then skips `base`.

**Gate.** This is a "leak-onset" one-neuron gate, fitted by the unchanged generic `fit_gate`. The task only defines
`probe_sites` and `gate_targets`.
- **Positives (`decision`, label 1):** the reply positions of de/ru/th practice base replies whose next token is a
  CJK token and whose own token isn't. That is, the onset of each leaked word.
- **Content negatives:**
  - every other position before the first leak in a leaking reply
  - every position in non-leaking de/ru/th replies
  - every position in zh and en replies, including zh positions that predict Chinese
  - this uses every 2nd position, to save memory
  - plus patchkit's 100 generic no_robots calibration replies (R1's 128-token English "thinking" openings). They
    are reasonable "ordinary text": the gate has to stay silent on English, and it reads only the residual, so a
    truncated reply is just more English positions.
- **Decision negatives:** 4 sampled positions per item, used only for `signal`'s AUC. They are also content
  negatives.
- **Write vector:** the generic one.
  - It lowers the 8 most frequent leaked onset tokens, weighted by count.
  - It raises `gate_targets`, which is read from `runs/langmix/gate_targets.json`. That file is computed in `data`
    on practice only. At each leak onset it takes the base model's argmax over non-CJK tokens, i.e. what the model
    says when Chinese is masked out. It keeps tokens that round-trip through `target_ids`.
  - **Only the top one is used** (`n_gate_targets = 1`): " ", which occurs at 5 of the onsets.
  - The next targets are script-specific: "ค" (Thai), " С", " доллар" (Russian), " dollars". Raising them in
    every language's reply would write foreign script. This was decided on practice data, before any held-out eval.
  - The targets are fixed from all of practice, including in the sweep. That's the same as entity's fixed "Sorry".
- **Hyper-parameters:** gain, g, k, margin and layer (-1) are patchkit's defaults.

**LoRA.** The fixed patchkit recipe, plus patchkit's calibration replay.
- **Positives:** trained on the reply generated with every CJK token banned by a logits processor. That's the cheap
  practitioner way to get a leak-free reply. Stored in `runs/langmix/ban_practice.jsonl`.
- **Negatives:** trained on their base reply.
- **Caveats:**
  - train_lora truncates each example to 768 tokens.
  - The replay pairs are 128-token *truncated* R1 replies followed by end-of-turn. That could teach early stopping
    on chat prompts; check `answered_*`.

**Prompt arm.** The old instruction arm: "Reply in the same language as the question." It's appended to the user turn
rather than put in a system prompt, because DeepSeek advises against system prompts for R1. So `render` is overridden.

**Recipe arm** (`gate_recipe`): the blind agents' continuous, context-gated penalty, recomputed from this task's
practice replies with the old fixed alpha = 0.25 and k = 256. See the module docstring of `tasks/_langmix_recipe.py`.

## Deviations from the old run

1. **Greedy instead of sampling (T=0.6, top-p 0.95), and 1,024 instead of 2,048 new tokens.**
   - Pilots on practice-pool items, 144 de/ru/th, 1,024 tokens:

     | Decoding | Leak de / ru / th | Unfinished |
     |---|---|---|
     | greedy | 6 / 25 / 19% | 22% |
     | sampled | 10 / 25 / 31% | 22% |

     So greedy keeps the failure.
   - In the old run's held-out base outputs, every leak occurs within the first 1,024 tokens: leak-by-token-1,024
     equals leak-by-2,048 for de, ru and th. So 1,024 tokens loses no leaks.
   - It does cut accuracy on long replies. Thai and Chinese are hit hardest: 42% and 16% of their old replies ran
     past 2,048 tokens.
   - Absolute numbers aren't comparable with the old table. The comparison that holds is the arm-vs-base deltas.
2. **No answer-format suffix.** My first pilot appended "Please reason step by step...". The model then thinks in
   English and barely leaks (1/120 greedy, 4% sampled). The old run's MGSM prompts had no suffix, so this port has
   none either.
3. **Held-out en and zh are 60 each, not 120,** to fit the runtime. de/ru/th stay at 120.
4. **Guards.**
   - IFEval is dropped. MMLU-200 and chat KL are patchkit's standard guards.
   - The old KL was measured on 256-token sampled answers.
5. **Practice.**
   - This port uses GSM8K-train items with greedy replies, as above.
   - The old agents used a mined practice set of sampled replies.
   - The recipe here is fitted on this practice data, so it isn't the agents' exact edit.
6. **Compat shim (proposed core change).** `patchkit.llm.eot_id` raises StopIteration for this tokenizer: it has
   none of `<end_of_turn>`, `<|im_end|>` or `<|eot_id|>`, and ends turns with its eos. `tasks/langmix.py` wraps it
   to fall back to `eos_token_id`. That changes nothing for the other tokenizers.
7. **Don't run `rescore langmix`.** patchkit saves only the first 1,500 characters of each output, and R1 replies are
   longer than that. Leaks and `\boxed{}` answers often come later, so rescoring the saved outputs would give wrong
   numbers. The full held-out texts are in `runs/langmix/full_outputs/` (one file per scored arm, in eval order).

## Examples (held-out, truncated around the leak; base vs fix)

1. **German, LoRA** (`mgsm-de-142`)
   - Base: "Zuerst bestimme ich die Gesamtzeit der**一周**, die Jordan hat."
   - LoRA: "Zuerst bestimme ich die Gesamtzeit der **Woche**, die Jordan hat zur Verarbeitung von Videospielen…"
2. **German, LoRA** (`mgsm-de-105`)
   - Base: "Dann addiere ich die Kekse, die**两国** insgesamt consumptionen: 15 Kekse für Cody…"
   - LoRA: "Nun berechne ich die Gesamtzahl der Kekse, die beide zusammen essen."
3. **Russian, recipe** (`mgsm-ru-32`)
   - Base: "Таким образом,**约翰** тратит 35 часов в неделю на уход за собаками."
   - Recipe: "Таким образом, **Джон** тратит 35 часов в неделю…"
   - The model's Chinese word for "John" becomes the Russian name.
4. **Russian, recipe** (`mgsm-ru-157`)
   - Base: "…и определить, кто более**长远**евший."
   - Recipe: "…и определить, кто более **бегает**."
5. **Thai, recipe** (`mgsm-th-116`)
   - Base: "…ที่คุณใช้น้ำตาลและน้ำที่คุณใช้ใน**比率** 7:13"
   - Recipe: "…ใน**อัตราส่วน**7:13", which is the Thai word for "ratio".
   - LoRA on the same item: "ให้ให้ให้ให้ให้…", a degenerate loop.
6. **Russian, recipe swaps Chinese for English** (`mgsm-ru-131`)
   - Base: "В мае он заплатил 500**美元**."
   - Recipe: "В мае он заплатил 500 **dollars**."
   - LoRA: "500 $"
7. **The one-neuron gate breaking the model** (`mgsm-de-65`)
   - Base: "5,5 Stunden/**周** × 4 Wochen = 22 Stunden"
   - Gate: "…al$maximal$maximal$maximal$maximal…", a loop until max_new.
   - In `mgsm-de-69` the gate writes "Bailey +=\nBailey +=\n…".
8. **The instruction prompt** (`mgsm-de-142`)
   - It makes the model think in English ("Okay, so I need to figure out how much money Jordan…").
   - That removes German leaks (de 9.2 → 2.5) but raises Russian ones (18.3 → 31.7).
