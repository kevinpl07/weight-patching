# Mid-training weight edit pilot

Does a direct LLM edit of a network's weights, made in the middle of gradient training, help the training that follows?

The design rule is the same as in "Sparks of Weight-Space Programming": the edit itself never goes through an optimizer. Training runs as normal AdamW. At a pause, an agent reads the weights and writes values into them with `edit_tools.py`, and then training resumes.

## Setup
- **Task:** (a + b) mod 97. All 9,409 pairs, 40% used for training.
- **Model:** a 2-layer ReLU MLP: 194 → 128 → 97, 37,473 weights.
- **Training:** full-batch AdamW, lr 1e-3, weight decay 1.0. The network memorizes the training pairs by about step 1,300 and groks (reaches 95% held-out accuracy) at 25k–33k steps.
- **Edit point:** 1,000 steps after training accuracy first reaches 100%, while held-out accuracy is still near 0.
- **Agent:** Opus 5.5 (`claude-opus-5-5`), given `runs/seedN/agent/PROMPT.md` word for word. It may change at most 2% of the weights (749). It can run forward passes on training pairs only, and every tool call is logged to `tool_log.jsonl`.
- **Arms:** each arm continues from the same checkpoint and keeps the same AdamW state. The arms are the unedited checkpoint, the agent's edit, and 3 random edits. Each random edit changes the same number of entries per tensor by the same total amount (L2 norm of the change), at random positions and in random directions.
- **Metric:** steps after the edit until held-out accuracy reaches 95% (and 50%).
- **Pass bar, set before running:** the agent's arm groks at least 30% sooner than the unedited arm in at least 2 of 3 seeds, and the random edits don't.

## Run
```
python experiment.py tune --seeds 0 1 2   # optional: plain runs
python experiment.py prepare --seed 0     # checkpoint + agent session in runs/seed0/agent
# agent turn: give runs/seed0/agent/PROMPT.md to the model as its task
python experiment.py forks --seed 0
python experiment.py report
```

The agent turn needs only a shell tool and the session folder. The prompt says everything else, so the same `PROMPT.md` works for an API agent loop with a single bash/exec tool.
