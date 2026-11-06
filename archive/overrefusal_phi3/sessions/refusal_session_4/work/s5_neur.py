import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
D=torch.load("work/acts_base.pt"); R=D["R"].float()
D2=torch.load("work/acts_base2.pt"); MA=D2["MA"]
W=T.base_weights(); P=T.practice(); g=[p["group"] for p in P]
O=torch.tensor([x=="over_refused" for x in g]); C=torch.tensor([x=="complied" for x in g]); H=torch.tensor([x=="harmful" for x in g])
X=R[:,24]; r=X[O|H].mean(0)-X[C].mean(0); r=r/r.norm(); torch.save(r,"work/r24.pt")
allrows=[]
for l in range(10,28):
    Wd=W[f"model.layers.{l}.mlp.down_proj.weight"].float()  # [3072,8192]
    w=r@Wd  # [8192]
    a=MA[l].float()
    c=a*w  # [300,8192]
    dO=c[O].mean(0)-c[C].mean(0); dH=c[H].mean(0)-c[C].mean(0); cC=c[C].mean(0)
    for j in range(8192): pass
    allrows.append((l,dO,dH,cC,c))
    top=torch.topk(dO,5)
    print(l,"sum dO %.2f dH %.2f | top dO neurons"%(dO.sum(),dH.sum()), [(int(j),round(dO[j].item(),2),round(dH[j].item(),2)) for j in top.indices])
torch.save([(l,dO,dH,cC) for l,dO,dH,cC,c in allrows],"work/neur_stats.pt")
# concentration: how many neurons explain gap
dOall=torch.cat([x[1] for x in allrows]); dHall=torch.cat([x[2] for x in allrows])
s,_=torch.sort(dOall,descending=True); print("total pos dO", s[s>0].sum().item(), "top50", s[:50].sum().item(),"top200",s[:200].sum().item(),"top1000",s[:1000].sum().item())
print("corr dO,dH across neurons", torch.corrcoef(torch.stack([dOall,dHall]))[0,1].item())
