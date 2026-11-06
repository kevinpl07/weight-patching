# weight-patching

**Can you fix a specific failure in a frozen LLM by editing its weights directly, with no training?
Sometimes, yes: one neuron is enough. Should you? Read on.**

This repo holds the code, data and results behind the blog post **[Weight-space programming, tested on real bugs](https://kevinpl07.github.io/weight-patching/)** by [Kevin Bondzio](https://x.com/KevBondzio). It has two parts:

1. **`patchkit/`**, a small framework for "weight patches". It covers:
   - one observable failure per task;
   - five ways to fix it (nothing, a system prompt, a one-neuron weight edit, a matched random edit, a LoRA fine-tune);
   - the same held-out data and guards for all five.

   Adding a new failure means writing one file (see [docs/ADDING_A_TASK.md](docs/ADDING_A_TASK.md)).
2. **`archive/`**: the earlier experiments (grokking, RL, pretraining, ViT, ResNet, Whisper, …), with their agent
   transcripts and reports. Most of them ask what happens to a weight edit *when training continues*.

> **Note on model outputs about real people.** The made-up-names task also asks about real people (from PopQA). Small models often invent false and sometimes damaging facts about them, so those outputs have been removed from `results/entity*/` and replaced with a placeholder. All metrics were computed before removal and are unchanged; re-run `python -m patchkit.run all entity` to regenerate the raw text.

## The one-neuron gate in 30 seconds

Most practical failures we looked at have the same shape:
- **The model already knows** it's in the bad case. A linear probe on its activations separates "made-up entity" from
  "real entity" with AUC ≈ 0.99.
- **It acts on that knowledge wrongly anyway.** It invents a bio.

The gate connects the two:

```
read   : the probe direction d at layer L's MLP input       (gate_proj row of the least-used neuron)
fire   : only above a threshold that no ordinary token reaches (set on practice data)
write  : push the logits toward an opener after which the model finishes the job itself, e.g. "Sorry"
                                                             (down_proj column)
```

That's about 3 × d_model numbers (7,677 for Gemma-3-4B) changed out of 4 billion. Nothing is trained, and the gate's
probe is fit on a few hundred practice prompts.

## Quick start

```bash
pip install -r requirements.txt
python -m patchkit.run all entity          # data → base → signal → gate + random → LoRA → eval (≈1 h on one 24 GB GPU)
python -m patchkit.run sweep entity        # gate vs LoRA as practice data grows
python -m patchkit.report                  # results/SUMMARY.md + site/data/results.json
```

`--model <hf id>` runs any task on another model.

## Results at a glance

Held-out data uses new items *and* new phrasings. Every fix sees only practice data. "Unchanged" counts the outputs
that are byte-identical to the unpatched model's on inputs where nothing should change.

| task (model) | failure ↓: base → prompt/decoding fix → **one neuron** → LoRA (same data) | unchanged elsewhere: one neuron vs LoRA |
|---|---|---|
| `entity` (Gemma-3-4B): invents bios for made-up people/films/books | 98% → 7%¹ → **16%** → 30% | ordinary chat answers 98% vs 0% |
| `agent_injection` (Qwen3-4B): coding agent runs commands hidden in a web page | 26% → 12.5% → **2.5%** → 0% | genuine tool-call replies 100% vs 72% |
| `signoff` (Gemma-3-1B): ends answers with "Would you like me to…" | 59% → 29%² → **23%** (2.3% with `--theta-q 0.999`) → 8% | answer content 69% vs 2% |
| `whisper_silence` (Whisper-small): transcribes words in silence/noise/music | 84% → 50% / 7%³ → **0.3%** → 0.3% | speech transcripts 100% vs 75% |
| `langmix` (R1-Distill-1.5B): Chinese words leak into de/ru/th answers | 15% → 15% → 17%⁴ → 9% | (agent-found edit: 0.3% leak, 100% chat unchanged) |

¹ but correct facts about real people fall from 57% to 20%. ² answers get 43% shorter. ³ OpenAI default / tuned
decoding thresholds; tuned thresholds drop 2% of real speech. ⁴ the one-neuron template doesn't fit this failure: leaks
aren't one decision point. The edit blind agents found earlier (a context-gated penalty on the Chinese-token output
rows) does fit it.

Also in `results/`:
- **`entity/sweep`:** gate vs LoRA from 8 to 720 practice items, 3 seeds.
- **`entity/bigdata`:** LoRA with 3–6× more data reaches 8–12% fabrication; the gate stays at ~15%.
- **`entity/pareto`:** LoRA at 6 checkpoints vs the gate at 5 strengths.
- **`entity/persistence*`:** 10–20 steps of unrelated fine-tuning erase the gate. The LoRA fix lasts ~50–100 steps.
- **`entity__Qwen*`:** a transfer test. The Qwen models barely fabricate, so there's nothing to patch.
- **`entity/agent_sessions`:** blind Claude Opus sessions.

Full tables: [results/SUMMARY.md](results/SUMMARY.md). Per-task notes: `results/<task>/NOTES.md`.

## Tasks

| task | model | code |
|---|---|---|
| `entity` | Gemma-3-4B-it | `tasks/entity.py` (the reference task) |
| `agent_injection` | Qwen3-4B-Instruct-2507 | `tasks/agent_injection.py` |
| `signoff` | Gemma-3-1B-it | `tasks/signoff.py` |
| `whisper_silence` | Whisper-small | `tasks/whisper_silence.py` (own CLI, same output contract) |
| `langmix` | DeepSeek-R1-Distill-Qwen-1.5B | `tasks/langmix.py` |

Protocol and its limits: [docs/PROTOCOL.md](docs/PROTOCOL.md). Blind agent sessions:
[docs/AGENT_SESSIONS.md](docs/AGENT_SESSIONS.md). The blog post's text: [docs/ARTICLE.md](docs/ARTICLE.md).

## Safety note

The agent-injection task never executes anything. The model's tool calls are only regex-checked, and attacker hosts
use reserved `.example` domains.

## License

Code: MIT. Model weights are not redistributed; edits are rebuilt from the scripts. Datasets keep their own licenses
(PopQA, no_robots, MMLU, SQuAD, LibriSpeech, ESC-50 (CC BY-NC), FMA).
