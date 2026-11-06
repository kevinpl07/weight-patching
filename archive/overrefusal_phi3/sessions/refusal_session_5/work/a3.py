import torch, numpy as np
D=torch.load("work/acts_last.pt"); act=D["mlp_act"].float()
for l in range(13,22):
    a=act[:,l]; m=a.mean(0); s=a.std(0)
    score=m.abs()/(s+1e-3)
    idx=torch.topk(torch.where(m.abs()>1.0,score,torch.zeros_like(score)),5).indices
    print(l,[(j, round(m[j].item(),2), round(s[j].item(),3)) for j in idx.tolist()])
