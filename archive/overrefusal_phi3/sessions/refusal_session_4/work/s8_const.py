import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
D2=torch.load("work/acts_base2.pt"); MA=D2["MA"]
P=T.practice(); g=[p["group"] for p in P]
O=torch.tensor([x=="over_refused" for x in g]); C=torch.tensor([x=="complied" for x in g]); H=torch.tensor([x=="harmful" for x in g])
for l in range(14,24):
    a=MA[l].float(); mu=a.mean(0); sd=a.std(0); cv=sd/mu.abs()
    grp=(a[O].mean(0)-a[H].mean(0)).abs()/mu.abs()
    score=torch.where(mu.abs()>0.5, cv+grp, torch.full_like(cv,9))
    idx=torch.topk(-score,4).indices
    print(l,[(int(j),round(mu[j].item(),2),round(cv[j].item(),3),round(grp[j].item(),3)) for j in idx], "max|mu|",round(mu.abs().max().item(),2))
