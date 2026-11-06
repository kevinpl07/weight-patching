# Pretraining plateau experiment: setup and scripts

Small GPTs trained from scratch on one RTX 4070, used to test whether an agent's weight edit at a loss plateau gives a lasting head start. The results are in `../RESULTS.md` ("agent edits during small-LM pretraining") and the summary is in `../FINDINGS.md`.

## Scripts

| File | What it does |
|---|---|
| `data.py` | Tokenizes a dataset into `data*/train.bin` and `val.bin` (uint16, documents joined by `<\|eos\|>`=0) and trains a byte-level BPE tokenizer. `--dataset tinystories` (4,096 tokens) → `data/`, `fineweb` (FineWeb-Edu `sample-10BT`, streamed, 8,192 tokens) → `data_fw/`, `code` (codeparrot-clean Python, streamed, 8,192 tokens) → `data_code/` |
| `model.py` | GPT with pre-LayerNorm, learned absolute positions, an untied unembedding and an optional attention-only mode. `forward(idx, return_attn=True)` also returns the attention patterns |
| `train.py` | AdamW training with a probe every `--probe-every` steps (see below). Batch order depends only on `--data-seed` and the step number, so `--init <ckpt>` resumes on exactly the same batches. `--stop-at N` stops early while keeping the LR schedule aimed at `--steps` |
| `plot.py` | 6-panel plot for one or more runs: training loss, val loss, repeated-text loss, copy benefit, and induction and previous-token scores per layer |
| `summarize.py` | Plateau table per run: onset (copy benefit ≥ 0.25), t50, t90, final values |
| `agent_tools.py` | The agent's tools: `info`, `base_weights`, `train_windows`, `run` (loss, attention, residual stream, each head's output), `metrics`, `save_edit` (budget and scope checks), and `lookahead` (paired training continuation, off unless `AGENT_LOOKAHEADS` > 0) |
| `agent_session.py` | `prepare` builds a session folder from a checkpoint, with loss history and earlier notes; `apply` writes the edit into the full checkpoint, optimizer state included; `random` writes a same-size random edit |
| `prompt_followup.md` | Prompt template for later interventions in a run |
| `compare_arms.py` | Compares `arm_*` runs in a session: val loss, gap to the control, copy benefit, steps saved |
| `probe_understanding.py` | Memorization vs generalization probes for a code checkpoint |

## Probes logged during training

- **`val_loss`:** loss on fixed val windows that lie inside one document.
- **`rep1` / `rep2`:** loss on the first and second copies of a repeated real passage. Copy benefit = rep1 − rep2. The same pair is also logged on random tokens (`rrep*`).
- **`prev[l][h]` / `ind[l][h]`:** attention to the previous token, and attention from the second copy to the token after the earlier occurrence (the induction pattern).
- **`icl`:** loss late in the window minus early. It is misleading on TinyStories and code, where the first tokens of a document are boilerplate and easy to predict.

## Reproduce the main experiment

```
.venv/Scripts/python.exe pretrain/data.py --dataset code --vocab 8192 --docs 200000 --val-docs 4000
.venv/Scripts/python.exe pretrain/train.py --out runs/pretrain_code/attn2L_d512_s0 --data-dir pretrain/data_code --ctx 512 \
    --steps 15000 --layers 2 --attn-only --d 512 --heads 8 --ckpt-every 1000 --ckpt-steps 250,500,750,1000 --compile
.venv/Scripts/python.exe pretrain/agent_session.py prepare --run runs/pretrain_code/attn2L_d512_s0 --step 1000 \
    --session runs/pretrain_agent/<name>
# agent turn: give <session>/PROMPT.md to the model (see runs/pretrain_agent/code_d512_s0_step1000/PROMPT.md)
.venv/Scripts/python.exe pretrain/agent_session.py apply --session runs/pretrain_agent/<name>
.venv/Scripts/python.exe pretrain/train.py --init runs/pretrain_agent/<name>/edited_step_01000.pt \
    --out runs/pretrain_agent/<name>/arm_agent --data-dir pretrain/data_code --steps 15000 --ckpt-every 1000 --compile
.venv/Scripts/python.exe pretrain/compare_arms.py runs/pretrain_agent/<name>
```

**Speeds on the 4070** (15,000 steps, 512-token context, batch 64):

| Model | Time |
|---|---|
| 2L attention-only d256 | about 10 min |
| 2L attention-only d512 | about 14 min |
| 4L GPT d256 | about 16 min |
| 6L GPT d384 | about 27 min |

`torch.compile` (`triton-windows`) adds about 10%. On Windows, a stopped background shell doesn't kill its `python.exe` children, so check for leftover `pretrain/train.py` processes.
