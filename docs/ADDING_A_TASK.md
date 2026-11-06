# Adding a task (a new failure to patch)

A **task** is one model plus one observable failure, plus the data to see it. Write one file, `tasks/<name>.py`,
with a `Task` subclass decorated with `@register`. Every method (base, prompt fix, one-neuron gate, matched random
edit, LoRA) and every guard (MMLU-200, chat KL) then works on it unchanged. `tasks/entity.py` is the reference.

## The five things you write

```python
@register
class MyTask(Task):
    name = "mytask"
    model = "google/gemma-3-4b-it"           # any HF chat model with a gated MLP (Gemma-3, Qwen3, Llama-style)
    system_fix = "..."                       # the prompt-engineering baseline you'd honestly try first
    gate_targets = ["Sorry"]                 # the reply opener the gate should write ("<eot>" = end the turn)
    headline = ("failure_metric", "kept_metric")   # two keys of score(): lower-is-better, higher-is-better

    def build_items(self, ctx): ...          # list of dicts: id, split ("practice" / "heldout"), prompt, labels
    def positive(self, item): ...            # True = failure case, False = must stay unchanged, None = ignore
    def score(self, items, texts): ...       # dict of metrics on held-out replies
```

Rules of thumb that held across all our experiments:
- **Held-out must differ from practice in phrasing**, not just in items. Otherwise you measure memorised wording.
- **`positive()` defines the signal.** The gate can only route a distinction the model already computes. Run
  `signal` first: if no layer has a cross-validated probe AUC above ~0.95, stop. A gate won't work.
- **Pick `gate_targets` by forcing.** Force a few candidate openers on practice failures (e.g. with
  `llm.generate(..., prefix="Sorry")`) and keep the one after which the model finishes the job by itself.
- **Write a "kept" metric that would catch collateral damage.** Example: for made-up entities, it's whether facts
  about real entities survive.

## Optional overrides

- `messages(item)` -> `(turns, system)` for multi-turn prompts or a persona (`base_system`, `tools` for tool calling).
- `probe_sites(tok, item, base_text)` if the decision isn't at the reply's first token. The sign-off task, for
  example, reads at every reply position. Return `dict(seq, pos=[...], label, kind="decision"|"content")`.
  "content" sites are ordinary positions the gate must stay silent on; the threshold is set above all of them.
- `failed(item, base_text)`: does the base reply to a positive item actually show the failure? The gate suppresses
  only what the model says when it fails. Without this, a model that already handles most cases correctly gets its
  own correct openers suppressed (we hit exactly this on Qwen3-4B, which mostly abstains already).
- `sft_target(item, base_text, forced_text)` if the LoRA baseline needs other targets. The default gives the same
  information the gate gets: the label plus the target opener.

## Recipe options worth knowing

These options are off by default, and every reported main result used the defaults. We found them during the
release runs.

- `--silence-after-fire` (`task.silence_after_fire = True`) adds the model's own replies after the target opener as
  must-stay-silent positions. **Use it when you transfer a task to a new model**: without it, the Llama-3.2-3B gate
  kept firing at every token. It doesn't change the Gemma entity result.
- `--theta-q 0.999` (`task.theta_content_q`) sets the threshold at a quantile of ordinary positions instead of their
  maximum. The max rule gets stricter as practice data grows (sign-off: 22% at full data). The 0.999 quantile gave 1.6%
  on a 64-prompt subset and 2.3% on the main held-out (LoRA: 8.3%). It matters for reply-position gates like sign-off; on the entity task (a first-token gate whose threshold is set by the good-case negatives) it changes nothing (14.2% vs 15.5%). Check it on your own held-out data.
- `--gate-g auto` makes the neuron's off state truly zero when ordinary positions sit close to the threshold
  (langmix).

## Run it

```bash
python -m patchkit.run data   mytask      # build items
python -m patchkit.run base   mytask      # base replies (+ forced-opener replies) on practice
python -m patchkit.run signal mytask      # stop-or-go: probe AUC per layer
python -m patchkit.run all    mytask      # signal, fit gate + random, train LoRA, eval all five arms
python -m patchkit.run sweep  mytask --ns 8,32,128,all --seeds 0,1,2    # gate vs LoRA as data grows
python -m patchkit.report                 # tables + site/data/results.json
```

`--model <hf id>` runs the same task on another model (results go to `results/<task>__<model>/`).

## Output contract

`results/<task>/<arm>.json` holds `headline`, `metrics`, `guards` (`mmlu200`, `chat_kl`) and `cost`
(`entries_changed`, `trainable_params`, `fit_seconds`, `n_practice_items`).
`results/<task>/outputs/<arm>.jsonl` holds `{id, text}` per held-out item.
Tasks that aren't chat LLMs (e.g. `tasks/whisper_silence.py`) implement their own commands but write the same
contract, so the report and the site treat them identically.
