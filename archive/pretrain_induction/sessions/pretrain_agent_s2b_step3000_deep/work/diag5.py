import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
x = T.train_windows(4, 512, seed=21)
o = T.run(x, edited=False, resid=True)
E = o["resid"][0]; R2=o["resid"][2]
pos = W["pos.weight"]; tokE = E - pos[None]
mu = R2.mean(-1,keepdim=True); sd = (R2-mu).pow(2).mean(-1,keepdim=True).add(1e-5).sqrt()
Wu = W["head.weight"]; lw = W["ln_f.weight"]
def lstd(v):
    z = (v - v.mean(-1,keepdim=True))/sd*lw
    lg = z.reshape(-1,512) @ Wu.T
    return lg.std(-1).mean().item()
comps = {"tok": tokE, "pos": pos[None].expand(4,-1,-1)}
for l in range(2):
    for h in range(8): comps[f"L{l}H{h}"] = o["head_out"][l][:,h]
for k,v in comps.items(): print(f"{k:6s} logit-std {lstd(v):.2f}")
print("total", lstd(R2))
print("sd mean", sd.mean().item())
