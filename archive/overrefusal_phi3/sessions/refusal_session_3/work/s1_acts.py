import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P=T.practice()
ids=[p["id"] for p in P]
out=T.teacher_forced(ids, layers=list(range(32)), what=("resid","mlp_act"), positions="last", last_logprobs=True)
R=torch.stack([torch.stack([o["acts"][("resid",l)][0] for l in range(32)]) for o in out])  # [300,33,3072]
M=torch.stack([torch.stack([o["acts"][("mlp_act",l)][0] for l in range(32)]) for o in out])  # [300,32,8192]
LP=torch.stack([o["last_logprobs"] for o in out])
torch.save(dict(ids=ids, groups=[p["group"] for p in P], R=R, M=M, LP=LP, top=[o["top_next"] for o in out]), "work/acts_last.pt")
for o in out[:3]+out[100:103]+out[200:203]: print(o["id"], o["top_next"][:5])
