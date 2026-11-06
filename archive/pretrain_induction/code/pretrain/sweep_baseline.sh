#!/bin/bash
# Baseline sweep: 3 seeds x {4-layer GPT, 2-layer attention-only}, 6000 steps each.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for s in 0 1 2; do
  $PY pretrain/train.py --out runs/pretrain/gpt4L_s$s --seed $s --steps 6000 --ckpt-every 500 > runs/pretrain/gpt4L_s$s.log 2>&1
  echo "finished gpt4L_s$s exit $?"
  $PY pretrain/train.py --out runs/pretrain/attn2L_s$s --seed $s --steps 6000 --ckpt-every 500 --layers 2 --attn-only > runs/pretrain/attn2L_s$s.log 2>&1
  echo "finished attn2L_s$s exit $?"
done
echo SWEEP_DONE
