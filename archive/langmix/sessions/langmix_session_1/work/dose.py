import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
torch.manual_seed(0)
G = torch.load("work/groups.pt", weights_only=False)["final"]
g = torch.load("work/gate_a025k256.pt", weights_only=False); us = g["us"]
W = T.base_weights()["lm_head.weight"].float()
cj = torch.zeros(W.shape[0], dtype=torch.bool); cj[torch.tensor(T.cjk_token_ids())] = True
for name in ["leak", "mixo", "clean", "zhc", "zho", "en"]:
    H = G[name]; H = H[torch.randperm(len(H))[:600]]
    L = H @ W.T; s = H @ us
    row = []
    for a in [0, 0.1, 0.25, 0.4]:
        La = L.clone(); La[:, cj] -= a * s[:, None]
        p = La.softmax(-1)[:, cj].sum(-1)
        row.append(f"a={a}: pcjk {p.mean():.4f}")
    print(name, " | ".join(row))
