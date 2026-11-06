import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P=T.practice(); ids=[p["id"] for p in P]
out=T.teacher_forced(ids, layers=list(range(33)), what=("resid",), positions="last")
R=torch.stack([torch.stack([o["acts"][("resid",l)][0] for l in range(33)]) for o in out])  # [300,33,3072]
torch.save(dict(ids=ids, resid=R, top=[o["top_next"][:5] for o in out], ntok=[len(o["ids"]) for o in out]), "work/resid_last.pt")
print(R.shape, sum(len(o["ids"]) for o in out))
for o in out[:3]+out[100:103]+out[200:203]: print(o["id"], o["top_next"][:4])
