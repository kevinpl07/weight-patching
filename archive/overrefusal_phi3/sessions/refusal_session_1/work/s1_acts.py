import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
sys.path.insert(0, "work"); from custom import SAFE, UNSAFE
import torch
s = T.info()
P = T.practice()
ids = [p["id"] for p in P]
items = ids + SAFE + UNSAFE
out = T.teacher_forced(items, layers=list(range(33)), what=("resid",), positions="last")
R = torch.stack([torch.stack([o["acts"][("resid", l)][0] for l in range(33)]) for o in out])  # [N,33,3072]
top = [o["top_next"][:5] for o in out]
torch.save(dict(R=R, top=top, items=items, toks=out[0]["tokens"]), "work/acts_last.pt")
print(R.shape, out[0]["tokens"][-8:])
for i in [0,100,200,300,340]: print(items[i] if isinstance(items[i],str) else i, top[i])
