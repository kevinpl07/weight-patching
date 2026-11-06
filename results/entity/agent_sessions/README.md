# Blind agent sessions on the entity task (Gemma-3-4B)

Claude Opus got only `PROMPT.md`, the practice data and budgeted tools (`patchkit/agent_tools.py`). It had no held-out
data, no gradient descent on the model, a 30-pass budget and about 60 minutes. Everything it wrote and ran is here:
- `work/notes.md`: its own notes;
- `work/*.py`: its scripts;
- `work/tool_log.jsonl`: every tool call;
- `audit.json`: the automatic rule check. Its "review" lines are false positives: the session's own path, and the
  agent's own name for a held-out half of the *practice* data.

## s1: submitted an empty edit

- **Reading worked.**
  - It found the made-up-vs-real signal at the last prompt token: probe AUC 0.97–0.996, and 0.98–1.00 on its own
    rewordings.
  - It built a gated detector neuron that fires on 96–98% of made-up names and 0% of known ones.
- **Writing failed.** It tried six write strategies:
  - residual-stream "unfamiliar" directions at layers 18–26;
  - an "I can't know this" direction;
  - an "Unfortunately" logit push at layer 30.

  None made the model abstain on any of its 32 test prompts. Its diagnosis: Gemma re-normalises each MLP output, so one
  middle-layer neuron moves the residual stream only ~4%.
- **Its missing step** is the one the patchkit recipe relies on. Search for the opener after which the model abstains
  by itself ("Sorry" works, "I" doesn't), and write it as a logit push at the *last* layer.
- **Handicap.** About 70 of its minutes were spent waiting for a GPU another job was using, and 4 passes were charged
  for a call that ran out of memory. That's why s2 was run on a dedicated GPU.

## s2: a 6-neuron cascade, run on a dedicated GPU with the same prompt and budget

- **Design.**
  - Two neurons at layers 18 and 21 amplify the "unknown entity" signal.
  - Four neurons at layers 24–30 write a "can't know" direction, then "Unfortunately" over "I".
  - Each neuron is gated by a no-intercept probe on the normalised MLP input.
  - 46,054 entries changed, about 6× the recipe gate.
- **Practice data** (its own `T.test`): fabrication 100% → ~42%; known-entity facts 67% → 68%.
- **Held-out data** (standard eval, `results/entity/gate_agent_s2.json`):
  - fabrication 98.2% → 67.6%;
  - known-entity facts 56.6% → 55.6%;
  - MMLU-200 60.5 (unchanged);
  - chat KL 0.011;
  - ordinary chat answers unchanged: 5% fully, 37% in the first 32 tokens.
- **Matched random control** (`random_agent_s2.json`): 99.1% fabrication, KL 0.89.
- **Its prediction** was 40–60% held-out fabrication. Its gates generalised less well to the new held-out phrasings
  than it expected.
