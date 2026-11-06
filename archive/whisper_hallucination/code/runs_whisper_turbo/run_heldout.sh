#!/bin/bash
# Transfer test: whisper-small recipe on openai/whisper-large-v3-turbo. Run from the repo root after the recipe build.
# All arms write to runs/whisper_turbo/eval/<arm> (never runs/whisper/eval). One model loaded at a time (sequential).
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 GPU_PRIORITY=-1
PY=.venv/Scripts/python
M=openai/whisper-large-v3-turbo
L=runs/whisper_turbo
B=${BATCH:-16}
E="$PY whisper_halluc/eval.py --model $M --out $L/eval --batch $B"
$E --arm turbo_base                                          > $L/eval_turbo_base.log 2>&1
$E --arm turbo_recipe     --edit $L/recipe/edit.safetensors       > $L/eval_turbo_recipe.log 2>&1
$E --arm turbo_vad        --vad                              > $L/eval_turbo_vad.log 2>&1
$E --arm turbo_recipe_vad --edit $L/recipe/edit.safetensors --vad > $L/eval_turbo_recipe_vad.log 2>&1
$PY whisper_halluc/session.py random --model $M --edit $L/recipe/edit.safetensors \
    --out $L/recipe/random0_count_edit.safetensors --mode count > $L/recipe/random.log 2>&1
$E --arm turbo_random_count --edit $L/recipe/random0_count_edit.safetensors > $L/eval_turbo_random_count.log 2>&1
echo ALL_DONE
