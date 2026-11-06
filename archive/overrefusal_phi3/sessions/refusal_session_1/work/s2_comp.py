import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
sys.path.insert(0, "work"); from custom import SAFE, UNSAFE
import torch
P = T.practice()
items = [p["id"] for p in P] + SAFE + UNSAFE
out = T.teacher_forced(items, layers=list(range(32)), what=("attn_out", "mlp_out", "mlp_act"), positions="last")
A = torch.stack([torch.stack([o["acts"][("attn_out", l)][0] for l in range(32)]) for o in out])
M = torch.stack([torch.stack([o["acts"][("mlp_out", l)][0] for l in range(32)]) for o in out])
Act = torch.stack([torch.stack([o["acts"][("mlp_act", l)][0] for l in range(32)]) for o in out])
torch.save(dict(A=A, M=M, Act=Act), "work/acts_comp.pt")
print(A.shape, Act.shape)
