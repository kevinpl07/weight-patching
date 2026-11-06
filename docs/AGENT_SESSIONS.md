# Blind agent sessions

An LLM agent (Claude Opus) gets only a session folder: a `PROMPT.md`, the task's **practice** data and budgeted
tools. It reads weights and activations and writes new values into the weights by hand to fix the task's failure.
No gradient descent touches the model. The edit is then scored by the standard eval (`patchkit.run eval`), next to a
matched random edit, exactly like the gate and LoRA arms. This is the protocol of the earlier blind sessions
(`RESULTS_2026-11-05_practical_edits.md`), ported to any patchkit task.

Files: `patchkit/agent_session.py` (prepare / apply / audit), `patchkit/agent_tools.py` (the tools),
`patchkit/agent_prompt_template.md` (the prompt).

## 1. Prepare

Needs `results/<task>/items.jsonl` and `runs/<task>/base_practice.jsonl` (`patchkit.run data` and `base`).
CPU only.

```bash
cd weight-patching
python -m patchkit.agent_session prepare entity --session runs/entity/agent/s1 \
    [--model google/gemma-3-1b-it] [--passes 30] [--budget-frac 0.001] [--minutes 60] [--examples 3]
```

The session folder gets:

| path | what |
|---|---|
| `PROMPT.md` | the brief: the model, the failure (task `description`, `headline_labels`, an optional `agent_brief` attribute on the task, practice examples), scoring, rules, tools, budgets, what to submit, a request to predict held-out scores |
| `practice/items.jsonl` | **practice items only**: the task's fields, `prompt`, `label` (1 / 0 / None from `positive()`), `base` (base greedy reply) |
| `practice/base_score.json` | `task.score` of the base replies on all practice items |
| `tools.py` | a shim: sets `AGENT_SESSION` and re-exports `patchkit.agent_tools` (Python API + CLI) |
| `session.json` | task, model, budgets |
| `work/` | the agent's scratch space: `notes.md`, `tool_log.jsonl`, `budget.json`, `clock.json`, `edited/edit.safetensors`, `edits/<tag>.safetensors`, `test_<k>.jsonl` |

Next to the folder, `<session>.manifest.json` holds hashes of the prepared files and of `agent_tools.py` for the
audit. Held-out items and the forced-opener replies (`forced` in `base_practice.jsonl`, which hint at the gate
method) are never copied.

**Budgets** (defaults): 30 passes, where a pass is one batch of up to 64 prompts or sequences for a counted tool call
(`generate`, `activations`, `logit_lens`, `unit_usage`, `test`); at most 0.1% of the language-model parameters
changed (vision towers are out of scope); about 60 minutes of wall clock (advisory, checked by `audit`).

## 2. Run the agent

Give the agent shell access to the session folder only (it needs the repo's `.venv` python, and read access to
`patchkit/agent_tools.py`), a working directory set to the session folder, and this instruction (fill in the path):

> Read `<session>/PROMPT.md` and do the task it describes. Work only inside `<session>`; outside it, read only the
> file PROMPT.md names as the tools' implementation. Don't look at any other file in the repository and don't use
> the internet. When you are done, reply with the report PROMPT.md asks for.

Opus with the default budgets needs about 60 minutes. The tools take the repo GPU lock themselves (priority 10, as an
agent label) for each counted call and release it after; **don't wrap the agent or its scripts in `gpu_lock.py`**
(that would deadlock: the inner calls wait for a lock their parent holds). The model loads once per Python process
onto the CPU (Gemma-3-4B: ~8.6 GB RAM, 10-20 s) and moves to the GPU only inside the lock. Edits are applied to the
GPU copy and undone before it moves back. Lock priority only matters when the lock changes hands: a long locked shell chain
(`gpu_lock.py -- bash -c "a && b && c"`) keeps the GPU for its whole length, so start sessions when no long chain
is running. Each counted call logs its lock wait (`gpu_wait_s`); `apply` and `audit` report the total, so waiting
can be told apart from the agent's own time.

Tool summary (full docstrings: `help(T.<name>)`, CLI: `python tools.py <cmd> --help`):

| free | counted (1 pass / 64 inputs) |
|---|---|
| `info`, `practice`, `prompt`, `tokenizer`, `weight_names`, `weight(name, rows, cols, edited)` | `generate(inputs, max_new, edited, prefix)` |
| `set_rows`, `set_cols`, `set_entries`, `set_tensor`, `clear_edit`, `current_edit`, `edit_summary` | `activations(inputs, layers, positions, kinds, edited)`: resid / attn_out / mlp_in / mlp_act / mlp_out at last_prompt / reply / all / explicit positions |
| `save_copy`, `load_copy`, `list_copies`, `budget` | `logit_lens(inputs, layers, k, watch)`, `unit_usage(inputs, layer)`, `test(ids, n=128)` (task.score on practice) |

The edit is stored sparsely in `work/edited/edit.safetensors` (`<name>::idx` flat int64 indices, `<name>::val` bf16
values); only entries whose bf16 value differs from the original count. Writes that would exceed the entry budget
are refused (and logged as `edit_rejected`).

## 3. Apply

```bash
python -m patchkit.agent_session apply entity --session runs/entity/agent/s1 [--model ...] [--tag agent_s1] [--seed 0]
```

Validates the submission (known in-scope tensors, indices, entry budget, passes), then writes

- `runs/<task>/edits/gate_<tag>.safetensors` + `.json`: the agent's edit as full tensors (the format
  `patchkit.llm.load_edit` / `make_arm` expect). Meta: `entries_changed`, `fit_seconds` (session wall time: first to
  last tool call), `n_items` (practice items), `passes_used`, `tool_calls`, per-tensor L2 of the change.
- `runs/<task>/edits/random_<tag>.safetensors` + `.json`: the matched random control. Same entries, same per-tensor L2
  norm of the change, Gaussian directions (entries that bf16 rounding would leave unchanged are nudged one step, so
  the count matches too).
- `<session>/submitted_edit.safetensors`: a copy of the sparse submission.

The default tag is `agent_<session folder name>`. The `gate_` prefix only reuses `make_arm`'s file convention; in the
results, `gate_agent_s1` is the agent's edit, not the one-neuron gate.

## 4. Evaluate

```bash
python -m patchkit.run eval entity [--model ...] --arms gate,random --tag agent_s1 [--no-mmlu] [--split heldout]
```

writes `results/<task>/gate_agent_s1.json` and `random_agent_s1.json` (headline, metrics, MMLU-200, chat KL, cost),
which `patchkit.report` picks up with the other arms. Wrap it in `gpu_lock.py --` like any other eval.

## 5. Audit

```bash
python -m patchkit.agent_session audit entity --session runs/entity/agent/s1
```

prints and writes `<session>/audit.json`: tool calls by kind, passes used (budget file and log), wall minutes, entries
changed, rejected edits, the scripts in `work/`, and flags from scanning `work/` scripts:

- **violation**: loading the model or its files outside the tools (`AutoModel...`, `...ForCausalLM`, hub downloads),
  adapters / trainers (`peft`, `Trainer(`)
- **review**: optimizers, `.backward()` / autograd, `scipy.optimize` (allowed only for a probe on activations, never
  on the model: read the line), GPU use outside the tools, paths to `results/`, `runs/` or held-out data

plus files changed since prepare (hash check). Inline `python -c` commands are not in `work/`; the prompt asks the
agent to write scripts, and the agent's transcript is the other half of the audit.
