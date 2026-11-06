import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
x = T.train_windows(8, 512, seed=21)
o = T.run(x, edited=False, resid=True)
E = o["resid"][0]; R1 = o["resid"][1]
pos = W["pos.weight"]; tokE = E - pos[None]
comps = {"tok": tokE, "pos": pos[None].expand(8,-1,-1)}
for h in range(8): comps[f"L0H{h}"] = o["head_out"][0][:,h]
mu = R1.mean(-1,keepdim=True); sd = (R1-mu).pow(2).mean(-1,keepdim=True).add(1e-5).sqrt()
lw = W["blocks.1.ln1.weight"]
qkv = W["blocks.1.attn.qkv.weight"]
def lin(v): return (v - v.mean(-1,keepdim=True))/sd*lw
full = lin(R1)
for hh in [2,1,4,6]:
    for nm, off in [("Q",0),("K",512),("V",1024)]:
        Wm = qkv[off+hh*64: off+hh*64+64]
        tot = (full @ Wm.T); tot = tot - tot.mean((0,1))
        T2 = tot.pow(2).sum(-1).mean()
        sh = {}
        for k,v in comps.items():
            c = lin(v) @ Wm.T; c = c - c.mean((0,1))
            sh[k] = ((c*tot).sum(-1).mean()/T2).item()
        print(f"L1H{hh} {nm}: " + " ".join(f"{k}:{v:.2f}" for k,v in sh.items()))
