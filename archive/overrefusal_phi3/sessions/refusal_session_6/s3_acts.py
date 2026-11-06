import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P=T.practice(); ids=[p["id"] for p in P]
Ls=list(range(6,30))
out=T.teacher_forced(ids, layers=Ls, what=("mlp_act","attn_heads"), positions="last")
A=torch.stack([torch.stack([o["acts"][("mlp_act",l)][0] for l in Ls]) for o in out])
Hh=torch.stack([torch.stack([o["acts"][("attn_heads",l)][0].reshape(-1) for l in Ls]) for o in out])
torch.save(dict(layers=Ls, mlp=A, heads=Hh), "work/mlp_heads_last.pt")
print("ok", A.shape, Hh.shape)
