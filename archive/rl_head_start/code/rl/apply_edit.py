"""Turn an agent's edit (safetensors of full replacement tensors) into a loadable model folder, optionally as a
matched random control.

    python rl/apply_edit.py runs/agent_dry/work/edited/edit.safetensors runs/agent_dry/model_edit
    python rl/apply_edit.py runs/agent_dry/work/edited/edit.safetensors runs/agent_dry/model_random --random --seed 0

--random keeps each tensor's set of changed entries and the size (L2 norm) of its change, but replaces the
change with Gaussian noise of that norm on the same entries.
"""
import argparse

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-0.8B"

p = argparse.ArgumentParser()
p.add_argument("edit")
p.add_argument("out")
p.add_argument("--random", action="store_true")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
sd = model.state_dict()
g = torch.Generator().manual_seed(a.seed)
for name, new in load_file(a.edit).items():
    base = sd[name].float()
    delta = new.float() - base
    if a.random:
        mask = delta != 0
        r = torch.randn(delta.shape, generator=g) * mask
        delta = r * (delta.norm() / r.norm())
    sd[name].copy_((base + delta).to(torch.bfloat16))
    print(f"{name}: {int((delta != 0).sum()):,} entries, |delta| = {delta.norm():.3f}")
model.save_pretrained(a.out)
print("saved", a.out)
