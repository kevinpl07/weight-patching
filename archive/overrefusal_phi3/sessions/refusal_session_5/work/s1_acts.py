import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P = T.practice()
ids = [p["id"] for p in P]
out = T.teacher_forced(ids, layers=None, what=("resid","attn_out","mlp_out","mlp_act"), positions="last")
D = {}
for k in ["resid","attn_out","mlp_out","mlp_act"]:
    D[k] = torch.stack([torch.stack([o["acts"][(k,l)][0] for l in range(32)]) for o in out])
D["ids"]=ids; D["groups"]=[p["group"] for p in P]
D["top_next"]=[o["top_next"][:5] for o in out]
D["tokens_example"]=out[0]["tokens"][-8:]
torch.save(D, "work/acts_last.pt")
for o in out[:3]+out[100:103]+out[200:203]: print(o["id"], o["top_next"][:4])
