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

## s2

See `s2/` (run on a dedicated GPU, same prompt and budget).
