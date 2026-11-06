# Task: fix a language model's failure by editing its weights directly

You are working on **{MODEL_NAME}**, a pretrained open chat model. It has a failure, in one line: **{DESCRIPTION}.** You will read its weights and activations and write new values into them directly. **Your edit is final.** No training happens afterwards, and the edited checkpoint is what gets scored, with the same prompts and decoding as today.

There is no training on your side. You may not compute gradients of the model, run an optimizer or gradient descent on its weights, train adapters (LoRA or anything else) or fit weights by regression or search against a loss. You *may* compute statistics from weights and activations and build an edit from what you find: means, differences of means, projections, norms, PCA/SVD, per-neuron activation profiles, the logit lens, and **linear probes on activations** (e.g. a logistic regression or least-squares direction fitted to activations you collected; scikit-learn or a few lines of numpy/torch are fine). The rule is about the model's weights: what you write into them must come from your own reasoning and from such statistics, not from an optimizer run on the model. Comparing a handful of hand-chosen variants, for example to check dose-response, is fine. Automated search loops over edit parameters or over model components are not.

## The model

- {MODEL_DESC} `T.info()` prints every tensor name and shape.
- **Usage, exactly as scored:** each prompt is one user turn in the model's chat template{SYSTEM_NOTE}. Greedy decoding, up to {MAX_NEW} new tokens. `T.prompt(id_or_text)` gives you the exact prompt string.

## The failure you are asked to fix

{FAILURE_TEXT}

The scored numbers are:
- **{H0_LABEL}** (`{H0}`, lower is better): the failure.
- **{H1_LABEL}** (`{H1}`, higher is better): the behaviour that must be kept.

## Your practice data

`T.practice()` returns {N_PRACTICE} practice items. Each has the task's own fields, the user message (`prompt`), the original model's greedy reply as scored (`base`) and a `label`:
- **label 1, {N_POS} items: failure cases.** The model's behaviour here should change.
- **label 0, {N_NEG} items: keepers.** Cases where the task wants today's behaviour kept (see the second headline number below); your edit must not make them worse.
- **label None, {N_NONE} items: unlabelled.** Part of the scoring population, but neither a clear failure nor a clear keeper.

The original model's scores on the whole practice set (from the replies in `base`; also in `practice/base_score.json`):

```
{BASE_SCORES}
```

Some practice items (replies cut to 300 characters):

{EXAMPLES}

Read more of them before you form a theory (`T.practice()`, or `practice/items.jsonl`).

## How you will be scored (you will not see this data)

- **Held-out items:** new items of the same kinds, **phrased differently from the practice prompts**. Your edit must work on the underlying situation, not on the practice wording or names. The headline numbers above (and the other task metrics: {OTHER_METRICS}) are measured on them with greedy decoding as above.
- **Collateral damage:** accuracy on 200 MMLU questions, and the KL divergence from the original model's next-token distributions on ordinary chat prompts (teacher-forced on the original model's replies).
- **Cost:** the number of weight entries you changed.
- **Controls, scored the same way:** the original model; a system-prompt fix; a random edit that changes exactly your entries by the same amount in random directions; and other methods built from the same practice data.

The question is whether careful reading and editing of the weights fixes the failure on new prompts without training and without collateral damage.

## Tools

Everything you need is in this folder. Run commands from it:

```
cd {SESSION}
PYTHONIOENCODING=utf-8 {PY} tools.py info
```

`tools.py` has a command line (`{PY} tools.py --help`; each command has `--help`) and a Python API. For anything beyond a quick look, write scripts in `work/` and run them from this folder (`{PY} work/look.py`). Start each script with:

```
import sys; sys.path.insert(0, "{SESSION}"); import tools as T
```

The model loads once per Python process (a few seconds to ~20 s), so put several tool calls in one script rather than running many tiny ones. `help(T.<function>)` documents every argument.

- **Free:**
  - `T.info()`: model summary, tensor names and shapes, special tokens, budgets
  - `T.practice(label="any")`: the practice items; `T.prompt(x)`: the exact prompt string; `T.tokenizer()`
  - `T.weight_names(regex)`, `T.weight(name, rows=..., cols=..., edited=False)`: read weights (float32 CPU copies)
  - editing (see below), `T.edit_summary()`, `T.budget()`
- **Counted, 1 pass per 64 prompts or sequences** (budget: **{PASSES} passes** in total):
  - `T.generate(inputs, max_new=None, edited=True, prefix=None)`: greedy replies as scored. `inputs` are practice ids, your own user messages (strings), `{"raw": exact_prompt_string}`.
  - `T.activations(inputs, layers, positions="last_prompt", kinds=("resid",), edited=True)`: teacher-forced activations at chosen positions: `"last_prompt"` (the position where the first reply token is decided), `"reply"` (every reply token), `"all"`, or explicit indices. A practice id is teacher-forced with the original model's own reply; pass `{"id": i, "reply": text}` or `{"user": text, "reply": text}` to force a reply of your choice. Kinds: `resid` (residual stream after the layer), `attn_out`, `mlp_in` (normalized MLP input), `mlp_act` (MLP hidden activation, the input of `down_proj`), `mlp_out`. The result's `rows` lists, for every row, the input, the position, the token there and the next token (the one the activation at that position helps predict).
  - `T.logit_lens(inputs, layers=None, k=10, watch=[...])`: top next-token predictions read from the residual stream after each layer, and the real output; `watch` tracks the log-prob of chosen tokens.
  - `T.unit_usage(inputs, layer)`: per-neuron MLP activation statistics (mean |activation|, mean, max) over all tokens.
  - `T.test(ids=None, n=128)`: the task's own metrics, computed exactly as scored, on practice items generated with your current edit, next to the original model's numbers on the same items. Default: a fixed sample of 128 labelled practice items (2 passes); `n="all"` for every practice item. Replies go to `work/test_<k>.jsonl`.
- `edited=True` means "with the current edit" (identical to the original model while no edit is saved); `edited=False` runs the original model.
- The GPU is shared with other jobs; each counted call waits for its turn, so a call can take a few minutes. Run scripts that make counted calls with a long shell timeout (10 minutes) so they aren't killed midway. A pass is charged only once the GPU work starts. Cache what you reuse (`torch.save` into `work/`) instead of recomputing it.

### Writing the edit

The current edit lives in `work/edited/edit.safetensors` and is updated by these calls (all free, all validated):

- `T.set_rows(name, rows, values)`, `T.set_cols(name, cols, values)`, `T.set_entries(name, index, values)`, `T.set_tensor(name, new)`: write new values into a tensor. They accumulate: each call changes one tensor of the current edit, and only entries that end up different from the original count.
- `T.clear_edit(name=None)`, `T.save_copy(tag)`, `T.load_copy(tag)`, `T.list_copies()`, `T.current_edit()` (`{name: full new tensor}`), `T.edit_summary()`.

Values are stored in bfloat16, like the model. A linear weight is `[out, in]`: a column of `down_proj` is one MLP neuron's output vector, a row of `gate_proj`/`up_proj` is what that neuron reads. Weights are only ever changed through these calls; don't load the model yourself.

## Rules

- **Edit budget:** at most **{MAX_CHANGED}** weight entries may differ from the original ({BUDGET_PCT} of the model's {N_PARAMS} parameters). {SCOPE} Smaller is fine.
- **No training:** no gradients of the model, no optimizer on its weights, no adapters, no fitting weights by regression or search against a loss (see above). Linear probes and other statistics on activations are fine.
- **Only the tools touch the model.** Don't load the model or its files yourself, and don't start GPU work outside the tools.
- **Show your evidence.** Explain:
  - where in the network the failure is decided, and what signal there tells failure cases from keepers
  - why your edit changes the failure cases and why it should leave the keepers (and ordinary chat) alone
  - what the edit does on practice data (`T.test`), next to the original model

  Check dose-response. The practice set is small and the held-out prompts are phrased differently: don't overfit practice wording.
- **Stay in your lane.** Work only in this folder. Outside it, you may read only `{AGENT_TOOLS}` (the implementation behind `tools.py`). Don't open other files or folders in the repository (no other `runs/` or `results/` folders, task definitions, evaluation code or notes), and don't use the internet.
- **Keep notes.** Keep `work/notes.md` up to date in plain English: what you looked at, what you concluded, and why. Keep copies of promising edits (`T.save_copy`). Write your code as scripts in `work/` (not long inline `python -c` commands) so the session can be audited. Don't print huge tensors. Every tool call is logged in `work/tool_log.jsonl`.
- **Time:** you have about **{MINUTES} minutes** and about 80 tool calls. Make sure a sensible edit is saved well before the end.

## Submission

Whatever is in `work/edited/edit.safetensors` at the end is your submission. It may be empty if you find no edit you trust. Your final reply should cover:

1. how you approached the problem, step by step
2. where the failure is decided in the network and what signal you found, with the evidence
3. the edit: which tensors, how many entries, and the norm of the change
4. its measured effect on practice data (`T.test`): `{H0}` and `{H1}` edited vs original, and the other metrics
5. your prediction for the held-out scores, with how confident you are in each:
   - `{H0}` ({H0_LABEL}) for the original model and with your edit
   - `{H1}` ({H1_LABEL}) for the original model and with your edit
   - the change in MMLU-200 accuracy and the chat KL divergence you expect
   - how a random edit of the same entries and size would do
