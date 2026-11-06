# langmix: experiment 2, language mixing (de/ru/th framing)

**Framing (user decision, 2026-11-04):** base DeepSeek-R1-Distill-Qwen-1.5B leaks Chinese into German, Russian and Thai
answers (check: `langmix/check_multi.py`, Han rate de 2/16, ru 4/16, th 5/16). Can a weight edit stop the leak while
Chinese prompts still get Chinese, English stays English, and accuracy holds? The first framing (English MATH-500)
failed its check (0/200 CJK); its suites are still buildable (`math500`, `gsm_sym`) but not in the defaults.

All commands run from the repo root with `PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
Every GPU use goes through `gpu_lock.py`, one lock per batch of 24, model on the CPU between batches,
`set_per_process_memory_fraction(0.85)` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Every script takes `--model` (hub id or local dir; agent tools: `AGENT_MODEL`). Model-dependent caches are keyed by the
last path component, as in the checks; use a separate `--out-root` / `--out` per model.

## Files

| file | what |
|---|---|
| `check.py`, `check_multi.py` | the orchestrator's viability checks (not modified; constants imported, rows reused by the base arm) |
| `common.py` | prompts, model loading, `apply_edit` (the single edit-application helper), CJK token list + ban processor, steering hook, `generate_chunk` (one locked batch) |
| `metrics.py` | mixing flags, per-language fields, the Chinese-reply rule, numeric and MATH scoring |
| `data.py` | fixed suites and practice pools, cached in `runs/langmix/data/*.json` (built) |
| `ifeval_check.py` | lm-eval's IFEval checker without the harness |
| `mine_practice.py` | the agent's practice sets (mix / clean / zh / en) |
| `eval.py` | scores one arm; resumable; `runs/langmix/eval/<arm>/{arm.json,<suite>.jsonl,summary.json}` |
| `agent_tools.py` | the agent's budgeted tools (`work/tool_log.jsonl`, `work/budget.json`, `work/edited/edit.safetensors`) |
| `session.py` | `prepare` / `apply` / `random` (matched random edits, `same` and `count` modes) |
| `prompt_template.md` | the blind agent prompt (`session.py prepare` fills in paths, model description, budget) |
| `baselines.py` | CJK ban check, steering vector, the baseline eval commands (`list`) |
| `summarize.py` | the pass-bar table, with 95% Wilson intervals for the base rates |

## Suites (data.py)

| suite | role | n | source |
|---|---|---|---|
| mgsm_de, mgsm_ru, mgsm_th | target | 120 each | juletxara/mgsm test; seed-0 shuffle of the 250 (check_multi.py's order), first 120. **Includes** the check's first 16 per language; the base arm imports those 48 rows |
| mgsm_en | retention | 120 | same 120 problems in English |
| mgsm_zh | retention | 120 | same 120 problems in Chinese |
| ifeval | collateral | 100 | google/IFEval, random subset (seed 0) |
| chat | collateral (KL) | 150 | HuggingFaceH4/no_robots *test*, English first user turns without a system message |
| practice pools | agent only | | de/ru/th/fr/es: Mathoctopus/GSM8KInstruct_Parallel (GSM8K *train* machine-translated, ungated); en: GSM8K train; zh: meta-math/GSM8K_zh train split |

Prompts: no system prompt, no instruction, the user turn is the question only (as check_multi.py). Sampling: T=0.6,
top-p 0.95, top-k off, 2,048 new tokens, chunks of 24, seed 1000 + index of the chunk's first item. IFEval and chat
prompts are used as is; the IFEval response scored is the text after `</think>`.

## Metrics (details in metrics.py's docstring)

- **mixing** (`mix`, primary): >= 1 Han character anywhere in reasoning + answer (for non-Han targets; also >= 3
  characters of another foreign non-Latin script). Also `han_run` (>= 2 consecutive Han), `han_answer` (Han after
  `</think>`), `think_lang_share` (share of langdetect-classified reasoning lines in the prompt's language),
  `think_script_share` (ru/th: share of reasoning letters in Cyrillic/Thai), `reply_target`, `lid_zh`.
- **Chinese reply** (`zh_reply`, MGSM-zh): the final answer after the last `</think>`, with LaTeX/boxed/math/digits
  removed, has >= 4 Han characters and at least as many Han characters as Latin words. No `</think>` = no answer =
  False (`has_answer`, `answer_en` reported).
- **accuracy**: last `\boxed{}` as a number, else the last number in the final answer (matches check_multi.py's
  scoring on all 96 of its rows).
- **chat KL**: mean per-token KL(base || arm) over the base model's cached 256-token sampled answers, each side under
  its own prompt and decoding mods (banned logits floored at log 1e-10).
- Language-specific logic lives only in `metrics.mix_flag`, `metrics.in_language`, `metrics.LANG_SCRIPT` and
  `common.SUFFIXES`; `--target-lang` forces a target language for every item.

## Command sequence

```bash
PY=".venv/Scripts/python"; export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1

$PY langmix/data.py                      # 0. suites (already cached)
$PY langmix/mine_practice.py             # 1. practice mining (RUN by the builder; resumable; --status)
for k in 1 2 3; do $PY langmix/session.py prepare --session runs/langmix/sessions/session_$k; done   # (done)
#    2. blind agent session k in runs/langmix/sessions/session_$k on PROMPT.md (about 40 min, 25 passes)
for k in 1 2 3; do
  $PY langmix/session.py apply  --session runs/langmix/sessions/session_$k
  $PY langmix/session.py random --session runs/langmix/sessions/session_$k --seed 0 --mode same
done
$PY langmix/baselines.py ban             # 3. baselines: sanity check of the CJK token list (CPU)
$PY langmix/eval.py --arm base           # 4. evals (base: done)
$PY langmix/run_chain.py                 #    one serial chain: instr, ban, then per session the agent arm (all
                                         #    suites) and its random `same` arm (mgsm_de/ru/th only), then summarize
$PY langmix/summarize.py                 # 5. table + verdicts
```

## Pass bar (summarize.py)

- **B1** de/ru/th mixing (pooled, 360 items) <= 1/3 of base (plain threshold).
- **B2** de+ru accuracy (pooled, 240 items; Thai is at the floor, 3%, so it stays in the mixing metric only),
  margin 3 points.
- **B3** MGSM-en accuracy, margin 3 points.
- **B4** MGSM-zh accuracy, margin 3 points, and Chinese-reply rate, margin 5 points.
- **B5** IFEval prompt-level strict, margin 2 points.
- **B6** every matched random edit fails one of B1-B5.

A session passes if all hold; the experiment succeeds at 2 of 3.

**Deviation (orchestrator's decision): B2-B5 are non-inferiority tests.** Bars of 2-5 points at n = 100-240 are inside
sampling noise: a no-op edit would fail one of them about 75-80% of the time. An arm fails B2-B5 only if its point
estimate is worse than base by more than the margin AND the drop is significant (one-sided unpaired two-proportion
z-test, p < 0.05). The raw differences are printed with 95% intervals. In practice, at n = 240 a drop must be about
7.5 points or more to fail B2. Random arms are scored on the mixing suites only, so they are judged on B1.

## Scope notes

- **Spanish:** a 16-prompt pilot on GSM8K-train translations leaked Chinese in 3/16 (19%; e.g. "se告诉我", "3000美元");
  French 0/16. Not added (scope fixed); a possible generalisation test for an edit
  (`runs/langmix/practice_pilot/gen.jsonl`).
- **Steering arm dropped** (optional in the plan; GPU cost). `baselines.py steer` and `eval.py --steer` remain
  available.

## Agent budget

25 counted passes (generate or teacher_forced); edit budget 0.5% of 1,777,088,000 parameters = **8,885,440 entries**
(`tie_word_embeddings=false`: `lm_head` and `model.norm` are in scope, input embeddings out). generate: sequences x
max_new <= 49,152 per side (24 x 2,048, ~3-4 min per side; `both=True` doubles it). teacher_forced: <= 32,768 tokens.

## GPU time

See the builder's report for measured numbers from the mining and the base arm; the README's earlier MATH estimates
no longer apply. A batch of 24 x 2,048 tokens takes 150-230 s, and a batch runs until its longest answer ends.
