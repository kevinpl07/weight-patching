import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P=T.practice(); ids=[p["id"] for p in P]
out=T.teacher_forced(ids, layers=list(range(32)), what=("resid","attn_out","mlp_out"), positions="last", edited=False, last_logprobs=True)
R=torch.stack([torch.stack([o["acts"][("resid",l)][0] for l in range(32)]) for o in out])
A=torch.stack([torch.stack([o["acts"][("attn_out",l)][0] for l in range(32)]) for o in out])
M=torch.stack([torch.stack([o["acts"][("mlp_out",l)][0] for l in range(32)]) for o in out])
LP=torch.stack([o["last_logprobs"] for o in out])
torch.save(dict(ids=ids, R=R, A=A, M=M, LP=LP, top=[o["top_next"] for o in out]), "work/acts_base.pt")
print("saved", R.shape, A.shape, M.shape, LP.shape)
