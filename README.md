# weight-patching

**Can you fix a specific failure in a frozen LLM by editing its weights directly, with no training?
Sometimes, yes: one neuron is enough. Should you? Read on.**

This repo holds the code, data and results behind the blog post *[TITLE]* ([site](site/index.html)). It has two parts:

1. **`patchkit/`**, a small framework for "weight patches". It covers:
   - one observable failure per task;
   - five ways to fix it (nothing, a system prompt, a one-neuron weight edit, a matched random edit, a LoRA fine-tune);
   - the same held-out data and guards for all five.

   Adding a new failure means writing one file (see [docs/ADDING_A_TASK.md](docs/ADDING_A_TASK.md)).
2. **`archive/`**: the earlier experiments (grokking, RL, pretraining, ViT, ResNet, Whisper, …), with their agent
   transcripts and reports. Most of them ask what happens to a weight edit *when training continues*.

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

## Tasks

| task | model | failure |
|---|---|---|
| `entity` | Gemma-3-4B-it | invents bios for made-up people, films and books |
| `agent_injection` | Qwen3-4B-Instruct | a coding agent runs shell commands hidden in a web page |
| `signoff` | Gemma-3-1B-it | ends answers with "Would you like me to…" sign-offs |
| `whisper_silence` | Whisper-small | transcribes words in silence, noise and music |

Results: [results/SUMMARY.md](results/SUMMARY.md). Protocol and its limits: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Safety note

The agent-injection task never executes anything. The model's tool calls are only regex-checked, and attacker hosts
use reserved `.example` domains.

## License

Code: MIT. Model weights are not redistributed; edits are rebuilt from the scripts. Datasets keep their own licenses
(PopQA, no_robots, MMLU, SQuAD, LibriSpeech, ESC-50 (CC BY-NC), FMA).
