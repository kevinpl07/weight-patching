import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P=T.practice(); ids=[p["id"] for p in P]; Ls=list(range(10,28))
out=T.teacher_forced(ids, layers=Ls, what=("mlp_act","attn_heads"), positions="last", edited=False)
MA={l:torch.stack([o["acts"][("mlp_act",l)][0] for o in out]) for l in Ls}
AH={l:torch.stack([o["acts"][("attn_heads",l)][0] for o in out]) for l in Ls}
torch.save(dict(MA=MA,AH=AH,layers=Ls),"work/acts_base2.pt"); print("ok", MA[10].shape, AH[10].shape)
