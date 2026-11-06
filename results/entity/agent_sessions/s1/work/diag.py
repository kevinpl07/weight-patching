# Does the neuron fire with the edit, and how much does it move the residual at the last prompt token?
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch, numpy as np
T.load_copy(sys.argv[1])
d = torch.load("work/acts1.pt"); ids = d["ids"][:12] + d["ids"][96:100]
A = T.activations(ids, [18, 22, 26, 30], kinds=("resid", "mlp_act", "mlp_out"))
for L, n in ((18, 4222), (22, 2086)):
    print(L, "neuron act", A[f"mlp_act@{L}"][:, n].numpy().round(2), "| mlp_out norm (pre-postnorm)", A[f"mlp_out@{L}"].norm(dim=1).numpy().round(0)[:6])
for L in (18, 22, 26, 30):
    base = torch.cat([d["acts"][f"resid@{L}"][:12], d["acts"][f"resid@{L}"][96:100]]) if f"resid@{L}" in d["acts"] else None
    if base is not None:
        print(L, "resid change norm", (A[f"resid@{L}"] - base).norm(dim=1).numpy().round(0), "resid norm", base.norm(dim=1).mean().item())
