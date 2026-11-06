# ViT locality experiment: setup and scripts

A small ViT trained from scratch on CIFAR-10 on one RTX 4070, used to test whether an agent's weight edit at
about 5% of training gives a head start that lasts to the final test accuracy. Plan, pass bar and predictions:
`../PLAN_2026-11-04_vit_locality.md`. Runs and agent sessions: `../runs/vision_vit/`.

## Scripts

| File | What it does |
|---|---|
| `data.py` | Downloads CIFAR-10 with torchvision into `data_cifar/raw/` and writes `data_cifar/cifar10_uint8.npz`. GPU-side loader: batch order and augmentation (4-pixel zero-pad random crop, horizontal flip) depend only on `(data_seed, step)` |
| `model.py` | ViT: patch 4 → 8×8 = 64 tokens, no class token (mean pool), learned `pos_embed`, pre-LN blocks with fused `qkv` (with bias). `forward(x, return_attn=True)` returns attention maps; `attn_stats` gives mean attention distance, 3×3-neighbourhood mass and self mass per head. `local_sigma` adds a fixed Gaussian locality bias (pilot reference only) |
| `train.py` | AdamW + warmup + cosine, bf16, optional `torch.compile`. Probes every `--probe-every` steps: full test set accuracy/loss, 5,000 fixed training images, attention stats on 500 test images. `--init <ckpt>` resumes model + optimizer + step on exactly the same batches; `--stop-at N` stops early with the LR schedule still aimed at `--steps`; `--mimetic0` applies the mimetic init at step 0 |
| `mimetic.py` | The scripted reference fix: mimetic initialization of attention (Trockman & Kolter 2023), usable at step 0 or added mid-training. The docstring states exactly what is done |
| `agent_tools.py` | The agent's tools: `info`, `base_weights`, `train_set`, `train_images`, `run` (logits, loss, attention, residual stream, per-head and per-MLP outputs), `metrics`, `save_edit` (budget and scope checks). Training images only |
| `agent_session.py` | `prepare` builds a session folder (base weights, `session.json`, `loss_so_far.csv`, `PROMPT.md` from `prompt_template.md`); `apply` writes the agent's edit into the full checkpoint; `random` a same-size random edit; `mimetic` the scripted fix; `none` a copy of the unedited checkpoint |
| `prompt_template.md` | The agent prompt, filled in per session by `prepare` |
| `run_arms.sh` | Queue of continued-training arms with at most `MAXJ` concurrent jobs |
| `data_hf.py` | CIFAR-10 from the Hugging Face parquet mirror (used here; the official server was too slow) |
| `compare.py` | Per-seed and mean-over-seed table of final / last-5 / span test accuracy, steps to fixed accuracy levels, attention distance, plus a 4-panel plot |

Data: `data_cifar/` holds downloaded files only (untrusted; nothing is executed from it).
`data_cifar/cifar10_train_only.npz` is the training split alone, the only data file agent sessions point to.

## Layout

- `runs/vision_vit/check/`: the recipe check (plain ×2, fixed locality bias ×1).
- `runs/vision_vit/base/s{0,1,2}/`: base runs to the edit step (step 200), checkpoint with optimizer state.
- `runs/vision_vit/s{seed}_step200/`: the agent's session folder (PROMPT.md, base weights, loss so far, `work/`). Nothing about test accuracy or the controls is in here.
- `runs/vision_vit/arms/s{seed}_step200/`: control checkpoints (`none_`, `mimetic_`, `edited_`, `random0_step_00200.pt`) and the continued-training arms `arm_*/` with their logs.

## Reproduce

```
.venv/Scripts/python.exe vision/data.py            # or vision/data_hf.py (what was used; see its docstring)
.venv/Scripts/python.exe vision/train.py --out runs/vision_vit/base/s0 --seed 0 --stop-at 200 --ckpt-steps 200 --compile
.venv/Scripts/python.exe vision/agent_session.py prepare --run runs/vision_vit/base/s0 --step 200 --session runs/vision_vit/s0_step200
.venv/Scripts/python.exe vision/agent_session.py none    --session runs/vision_vit/s0_step200
.venv/Scripts/python.exe vision/agent_session.py mimetic --session runs/vision_vit/s0_step200 --seed 0
# agent turn: give runs/vision_vit/s0_step200/PROMPT.md to a fresh model
.venv/Scripts/python.exe vision/agent_session.py apply   --session runs/vision_vit/s0_step200
.venv/Scripts/python.exe vision/agent_session.py random  --session runs/vision_vit/s0_step200 --seed 0
MAXJ=1 bash vision/run_arms.sh runs/vision_vit/arms/s0_step200 agent runs/vision_vit/arms/s0_step200 random0
.venv/Scripts/python.exe vision/compare.py "runs/vision_vit/arms/s*_step200"
```

Recipe defaults live in `train.py` (depth 6, d192, 3 heads, batch 512, 4,000 steps, warmup 200, lr 1e-3, wd 0.05).
Speed on the 4070: about 3 min per 4,000-step run alone (incl. about 40 s of compilation); the GPU is saturated by
one job, so concurrency doesn't add throughput, and each job takes about 3–3.5 GB. On Windows, a stopped background
shell doesn't kill its `python.exe` children (the venv launcher also spawns a second `python3.13.exe`).
