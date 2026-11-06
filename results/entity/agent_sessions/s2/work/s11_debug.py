# Step 11: debug why the unit is silent: recompute gate/up pre-activations offline from the stored mlp_in.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch
A = torch.load("work/s7_acts.pt", weights_only=False); nf = A["n"][0]
NE = {24: 525, 26: 662, 28: 1066, 30: 2524}
D = torch.load("work/s8_dirs.pt", weights_only=False)
for Lr, j in NE.items():
    g = T.weight(f"model.language_model.layers.{Lr}.mlp.gate_proj.weight", rows=[j], edited=True)[0]
    u = T.weight(f"model.language_model.layers.{Lr}.mlp.up_proj.weight", rows=[j], edited=True)[0]
    X = A[f"mlp_in@{Lr}"][:nf]
    w = torch.as_tensor(D[Lr]).float()
    print(Lr, "gate.x", (X @ g).median().item(), "up.x", (X @ u).median().item(), "w.x", (X @ w).median().item(), "|g|", g.norm().item(), "|w|", w.norm().item())
