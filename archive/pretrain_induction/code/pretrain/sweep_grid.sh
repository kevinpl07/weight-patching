#!/bin/bash
# Dataset x size grid, seed 0, 15k steps, ctx 512, batch 64. Runs one at a time.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
CK=$(seq -s, 250 250 3000)
run() {  # name data_dir out_root args...
  local name=$1 data=$2 root=$3; shift 3
  if [ -f $root/$name/ckpt/step_15000.pt ]; then echo "skip $root/$name"; return; fi
  $PY pretrain/train.py --out $root/$name --data-dir $data --ctx 512 --seed 0 --steps 15000 \
      --ckpt-every 1000 --ckpt-steps $CK --compile "$@" > $root/$name.log 2>&1
  echo "finished $root/$name exit $?"
}
mkdir -p runs/pretrain_code runs/pretrain_fw
run attn2L_s0       pretrain/data_code runs/pretrain_code --layers 2 --attn-only
run gpt4L_s0        pretrain/data_code runs/pretrain_code --layers 4
run attn2L_d512_s0  pretrain/data_code runs/pretrain_code --layers 2 --attn-only --d 512 --heads 8
run attn2L_d512_s0  pretrain/data_fw   runs/pretrain_fw   --layers 2 --attn-only --d 512 --heads 8
run gpt6L_d384_s0   pretrain/data_code runs/pretrain_code --layers 6 --d 384 --heads 6
run gpt6L_d384_s0   pretrain/data_fw   runs/pretrain_fw   --layers 6 --d 384 --heads 6
echo GRID_DONE
