import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch
W=T.base_weights()
for n in ["blocks.0.ln1.weight","blocks.1.ln1.weight","ln_f.weight"]:
    w=W[n]; print(n, "min",w.min().item(),"absmin",w.abs().min().item(),"q05",w.quantile(0.05).item(),"mean",w.mean().item(),"max",w.max().item())
d=torch.load("work/r1_seed2.pt")
print("sigma1 mean", d["r1"].std(-1).mean().item(), "sigma0", (W["tok.weight"][d["x"]]+W["pos.weight"][:512]).std(-1).mean().item())
