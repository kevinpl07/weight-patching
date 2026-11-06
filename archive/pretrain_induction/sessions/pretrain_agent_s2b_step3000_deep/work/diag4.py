import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
x = T.train_windows(16, 512, seed=21)
o = T.run(x, edited=False, resid=True, attn=True)
E = o["resid"][0]; R1 = o["resid"][1]; R2=o["resid"][2]
pos = W["pos.weight"]; tokE = E - pos[None]
def stats(name, v):
    flat = v.reshape(-1,512)
    mu = flat.mean(0)
    cen = flat - mu
    print(f"{name:10s} rms-norm {flat.norm(dim=-1).mean():6.2f} mean-vec norm {mu.norm():6.2f}  varying norm {cen.norm(dim=-1).mean():6.2f}")
stats("tok", tokE); stats("pos", pos[None].expand(16,-1,-1)); stats("embed", E)
for h in range(8): stats(f"L0H{h}", o["head_out"][0][:,h])
for h in range(8): stats(f"L1H{h}", o["head_out"][1][:,h])
stats("resid1", R1); stats("resid2", R2)
# how much of each varying component survives in LN (variance share of resid1 after centering per-token mean)
cen = lambda v: (v - v.mean(-1,keepdim=True))
tot = cen(R1).pow(2).sum(-1).mean()
comps = {"tok": tokE, "pos": pos[None].expand(16,-1,-1)}
for h in range(8): comps[f"L0H{h}"] = o["head_out"][0][:,h]
for k,v in comps.items():
    print(f"share of ||resid1||^2 projected on {k}: {((cen(v)*cen(R1)).sum(-1).mean()/tot).item():.3f}")
# L0H3 attention offset profile & what it attends to
A = o["attn"][0]
t = torch.arange(100, 512)
for h in [3,4,5]:
    prof = [A[:,h,t,t-k].mean().item() for k in range(0,10)]
    print(f"L0H{h} offsets 0-9:", " ".join(f"{v:.2f}" for v in prof))
A1 = o["attn"][1]
for h in range(8):
    prof = [A1[:,h,t,t-k].mean().item() for k in range(0,10)]
    print(f"L1H{h} offsets 0-9:", " ".join(f"{v:.2f}" for v in prof))
