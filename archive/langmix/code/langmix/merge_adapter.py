"""Merge a PEFT LoRA adapter into its base model on CPU and save a plain HF checkpoint.

    .venv/Scripts/python langmix/merge_adapter.py --adapter Tina-Yi/R1-Distill-Qwen-1.5B-Open-RS1 \
        --subfolder checkpoint-2400 --out runs/langmix/models/tina-rs1-2400
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--subfolder", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, a.adapter, subfolder=a.subfolder).merge_and_unload()
model.save_pretrained(a.out)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print("saved", a.out)
