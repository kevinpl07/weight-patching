# Experiment 1: over-refusal (PLAN_2026-11-04_practical_edits.md, section 1)

Code in `refusal/`, outputs in `runs/refusal/`. Every script takes `--model` (default `meta-llama/Llama-3.2-1B-Instruct`
or `$REFUSAL_MODEL`). Smoke-tested end to end on `google/gemma-3-1b-it`; checks run on Llama-3.2-1B/3B, Qwen3-0.6B,
Phi-3-mini, Phi-3.5-mini, Gemma-2-2B. All GPU work takes `gpu_lock.py` per chunk (about 1 min) and caps memory at
0.85 of the card. Batch sizes adapt to the KV-cache memory of the model (`common.auto_batch`), and base-vs-edited
comparisons swap only the edited tensors in place (`common.Swap`), so 3-4B models need one copy in RAM and on the GPU.

```
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python
M=microsoft/Phi-3-mini-4k-instruct           # the only candidate over the bar (see "Model choice")
S=runs/refusal/sessions/$(basename $M)
EV="--ifeval-limit 100 --ifeval-max-new 512 --chunk 32"   # fixed for every arm (orchestrator decision, 2026-11-05)
```

Decided: Phi-3-mini, no system prompt. Sessions: `$S/session_{1..6}`. **Short protocol (from 2026-11-05 04:20)**:

```
FULL="--suites xstest_safe,orhard,xstest_unsafe,ortoxic,mmlu,kl,ifeval --gate-ifeval --refusal-max-new 64 --ifeval-limit 100 --ifeval-max-new 512 --chunk 96"
RAND="--suites xstest_safe,orhard,xstest_unsafe,ortoxic --refusal-max-new 64 --chunk 96"
$PY refusal/eval.py --config <arm.json> $FULL     # baselines, agent and recipe arms
$PY refusal/eval.py --config <random arm.json> $RAND
```

- Refusal suites are generated and scored on the first **64** tokens (`--refusal-max-new 64`). The base arm was run at
  256 and re-judged on its first 64 tokens with `refusal/truncate.py --write` (originals kept as `*_256.jsonl`).
  Agreement of 64- vs 256-token labels on base (binary refusal, "any"): **100.0%** (950/950; also 100% at 32, 48,
  96 and 128 tokens; no 3-class label changes). Phi-3 refuses in its first few tokens. Caveat: the judge's
  partial-refusal window is 400 characters (~90 tokens), so an edited model that refuses only after a long preamble
  would be missed at 64; the LLM-judge spot check covers this.
- IFEval (first 100 prompts, 512 tokens) runs for base and for arms that pass c1 and c2 on the refusal suites
  (`--gate-ifeval`; the decision is stored as `ifeval_gate` in summary.json). MMLU-500 and KL run for every non-random
  arm. Random arms get the refusal suites only.
- Measured Phi-3 GPU minutes per arm (lock waits excluded): refusal suites at 64 tokens ~4-6, MMLU-500 ~9 (560 s on
  base), KL ~3, IFEval-100 at 512 tokens ~9 when gated in. Non-random arm ~16-18 (+9 with IFEval), random arm ~5.

## Command sequence

GPU minutes are measured for Gemma-3-1B (Llama-3.2-1B is ~2-3x faster; 3-4B models ~3-5x slower).

| # | step | command | GPU min |
|---|---|---|---|
| 1 | data splits (CPU, once for all models) | `$PY refusal/data.py prep` | 0 |
| 2 | viability check, XSTest-safe 250 | `$PY refusal/check.py --model $M` (optional `--system-preset llama2`) | 1.3 |
| 3 | label the practice pool, pick 100/100/100 | `$PY refusal/data.py label --model $M` | 4-8 |
| 4 | base arm, all suites | `$PY refusal/eval.py --model $M --arm base` | ~18 |
| 5 | prepare 3 sessions | `for s in s1 s2 s3; do $PY refusal/session.py prepare --model $M --session $S/$s; done` | 0 |
| 6 | blind sessions: agent reads `$S/sN/PROMPT.md`, works in `$S/sN` | (orchestrator) | <= 30 passes x 0.2-0.7 |
| 7 | collect edits + matched randoms | `$PY refusal/session.py apply --session $S/sN`; `$PY refusal/session.py random --session $S/sN --seed 0 --mode same` (optionally `--mode count`) | 0 |
| 8 | script arms | `$PY refusal/baselines.py all --model $M` | ~3 |
| 9 | evaluate arms | `$PY refusal/eval.py --config runs/refusal/arms/<m>/{sysprompt,abliterate,targeted}.json`; `$PY refusal/eval.py --config $S/sN/arm_agent_sN.json`; `$PY refusal/eval.py --config $S/sN/arm_random0_same_sN.json` | ~18 each |
| 10 | LLM-judge spot check | each arm dir has `judge_sample.jsonl` (20% of refusal-suite responses) and `judge_prompt.txt`; subagents write `judge_labels.jsonl` (`{"uid", "label"}` per line) next to it; then `$PY refusal/judge.py agree --sample <dir>/judge_sample.jsonl --labels <dir>/judge_labels.jsonl` | 0 |
| 11 | table + pass bar | `$PY refusal/summarize.py --model $M` (writes `runs/refusal/summary_<m>.md`) | 0 |
| 12 | recipe arm (after sessions) | a <=30-line script reproducing each agent's recipe saves an edit; `$PY refusal/eval.py --model $M --arm recipe_sN --edit <file>` | ~18 |

Arms: base, sysprompt, abliterate, targeted, agent x3, random_same x3 (+ random_count x3) = 10-13 arms, about
3-4 GPU hours on Gemma-3-1B. All eval runs are resumable (rerun the same command); `--suites` picks suites.

Phi-3-mini (3.8B, 32 KV heads, no GQA) is ~3.5x slower and its KV cache limits batches to ~9 at 256 tokens and ~2-3
at 1,024: XSTest-safe 250 took 262 s, OR-Hard 300 took 267 s, so the refusal suites are ~17 min per arm and IFEval at
1,024 tokens would be ~3 h per arm. Use `--ifeval-max-new 512` and/or `--ifeval-limit N` (the settings are recorded in
summary.json; keep them identical across arms).

Per-arm cost on Gemma-3-1B (batch 64): refusal suites 950 prompts x 256 tokens ~5 min (XSTest-safe 250 took 78 s,
OR-Hard 300 took 84 s); MMLU 500 ~0.5 min; IFEval 541 x 1,024 tokens ~10-12 min (the dominant cost; pass
`--suites` without ifeval for quick looks); KL ~1 min (+0.7 min once per model for the base continuations).

## Model choice (check results, greedy, no system prompt, string judge "any" = full + partial)

| model | XSTest-safe full / partial / any | OR-Hard held-out (300) full / partial / any |
|---|---|---|
| Qwen3-0.6B | 3.2 / 0.4 / 3.6 | - |
| Llama-3.2-1B-Instruct | 6.4 / 0.0 / 6.4 | 15.7 / 0.7 / 16.3 |
| Llama-3.2-3B-Instruct | 5.6 / 0.0 / 5.6 | 8.3 / 0.7 / 9.0 |
| gemma-3-1b-it | 10.4 / 1.2 / 11.6 | 20.0 / 16.0 / 36.0 |
| gemma-2-2b-it | 3.6 / 4.4 / 8.0 | 24.0 / 21.0 / 45.0 |
| Phi-3.5-mini-instruct | 1.2 / 0.0 / 1.2 (lectures missed, <= ~8) | 2.0 / 1.3 / 3.3 (lectures missed, up to ~38) |
| **Phi-3-mini-4k-instruct** | 25.2 / 0.4 / **25.6** | 45.0 / 0.0 / **45.0** |

Rows in `runs/refusal/eval/<model>/base/` (reused as the base arm), summaries in `runs/refusal/check/`.

## Files

- `common.py` loading (Gemma-2 in eager attention for its softcapping), chat formatting, generation with automatic
  batch size, `apply_edit` / `Swap` (the one edit-application path used by eval, sessions, agent tools, baselines),
  the tied-embedding rule, and the residual-writer structure incl. post-norms.
- `data.py` splits (`runs/refusal/data/held_out.json`, `practice_pool.json`) and practice labelling
  (`runs/refusal/data/<m>/practice.json`, `pool_labels.jsonl`).
- `judge.py` XSTest string-match judge (full / partial / compliance), 20% export, agreement with external labels.
- `check.py` viability check. `eval.py` arm evaluation. `summarize.py` pass-bar table. `truncate.py` re-judges an
  arm on the first N tokens of its stored answers (no GPU).
- `agent_tools.py` budgeted tools (30 passes, 1% of entries). `session.py` prepare / apply / random.
  `prompt_template.md` the blind agent prompt (filled per model by `session.py prepare`).
- `baselines.py` system-prompt arm, plain abliteration, targeted abliteration.

## Decisions and deviations

- **XSTest source:** `walledai/XSTest` is gated for this account; `Paul/XSTest` (the author's own upload, ungated)
  has the same 450 prompts (250 safe / 200 unsafe) with type labels.
- **Refusal metric:** primary = "any" (full + partial). Gemma models open most refusals with an empathy preamble ("I
  understand you're going through ... However, I cannot ..."), which XSTest's prefix matcher calls compliance, and
  answer many scary-sounding benign prompts with crisis counselling (counted as full refusal). `full` is reported
  alongside (`summarize.py --metric full`). Labels are recomputed from the stored text at summary time.
- **Tied embeddings:** edits may name `model.embed_tokens.weight` only; `lm_head.weight` is rejected for tied models
  (Gemma, Llama-3.2, Qwen3); its entries count once. Phi-3 is untied and its `lm_head.weight` is a separate parameter.
- **Post-norms (Gemma 2/3):** o_proj/down_proj outputs pass through an RMSNorm (gain g = 1+w, ~0 to ~200) before the
  residual add, so orthogonalizing them against r does not remove r (measured: |cos(write, r)| unchanged). Each
  writer's output is made orthogonal to g*r instead (exact removal; the RMS rescale is a per-token scalar), via a
  regularized oblique projection along r*g/(g^2 + 0.1 median g^2) so the rest of the write changes little.
  See `common.writer_projection`.
- **Massive activations:** a raw difference of means is dominated by the massive-activation channels (Gemma-3-1B dim
  1038, mean |x| ~3,100; 83% of the unit direction). Projecting that out of every writer destroys the model (KL ~20
  at every layer). Both abliteration arms zero coordinates with mean |x| > 10x the median before normalizing.
- **Fused projections (Phi-3):** `qkv_proj` and `gate_up_proj` are fused; the residual writers (`o_proj`, `down_proj`)
  and the hooks (input to `o_proj` = per-head outputs, input to `down_proj` = MLP activation) are unaffected.
  `T.info()` tells the agent about the fused layout.
- **System prompts:** templates without a system role (Gemma 2 raises, Gemma 3 folds) get the system text folded into
  the user turn as `<system>\n\n<user>`.
- **KL prompts:** `HuggingFaceH4/no_robots` test split (ungated, 11 MB), 200 prompts <= 2,000 characters.
- **Eval outputs** go to `runs/refusal/eval/<model>/<arm>/` (model level added so runs on different models never mix).
- **Practice "over-refused"** = OR-Hard prompts the base model refuses ("any"); labelling walks the 1,019 non-held-out
  OR-Hard prompts until it has 100. "Complied" prefers OR-Hard prompts the model answers, topped up from OR-Bench-80k.
- **Llama date:** the Llama 3 template inserts today's date; all prompts pin `date_string="26 Jul 2024"`.
- **Random controls** match each tensor's changed-entry set (or count) and L2 norm after bf16 rounding; entries that
  rounding would leave unchanged are moved one bf16 step, unless that alone overshoots the norm by >10% (tiny edits
  to large-valued norm gains), in which case the norm wins and the script prints the achieved count.
- **Abliteration layer pick:** Arditi's filter (first-token KL < 0.1 on held-in complied prompts) passed no layer on
  Gemma-3-1B in the smoke run (best 0.5 at layer 17); the rule then falls back to the lowest-KL layer. The sweep table
  is saved in `runs/refusal/arms/<m>/abliterate_meta.json`.
- **MMLU letters** are scored as logsumexp over each letter's token variants ("A", "▁A", " A"), since SentencePiece
  models (Phi-3) emit the bare letter after a newline.
- **Generate cap per agent call:** 256 prompts for models <= 2B params, 96 above (one pass of 96 x 256 tokens on
  Phi-3-mini is ~1.5 GPU min; written into session.json and the prompt).
- **GPU lock starvation:** with the language-mixing and Whisper jobs running, single refusal passes waited 5-15 min
  for the lock (filelock is not FIFO). Agent sessions and arm evals need a quiet GPU to fit their time budgets.
- **Smoke artefacts:** `runs/refusal/data/gemma-3-1b-it/practice.json` comes from a `--limit 120` smoke labelling;
  rerun `data.py label` without `--limit` (it resumes). The smoke baseline arms built from it were deleted.
  Smoke sessions live in `runs/refusal/sessions/*/smoke`.
- **Abliteration dose curves:** `baselines.py dose --alphas 0.2 0.4 0.6` writes `abliterate_a<alpha>` and
  `targeted_a<alpha>` arms: W' = W - alpha * (projection), same direction, layer choice/window and tensors as the full
  arms (embedding term scaled too). summarize.py adds a matched-false-refusal table: for each agent and recipe arm,
  the harmful-refusal rate the plain and targeted curves reach at that arm's FR (alpha = 0 base ... 1 full, linear
  between adjacent alphas) and the arm's advantage in points.
- **Recipe arms** (`runs/refusal/recipe/arm_recipe_*.json`, from `refusal/recipe_script.py`) are evaluated with the
  agent protocol and listed as kind "recipe".
