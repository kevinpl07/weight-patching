import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
D2=torch.load("work/acts_base2.pt"); MA=D2["MA"]
P=T.practice(); g=[p["group"] for p in P]
O=torch.tensor([x=="over_refused" for x in g]); C=torch.tensor([x=="complied" for x in g]); H=torch.tensor([x=="harmful" for x in g])
gen=torch.Generator().manual_seed(0); half=torch.rand(300,generator=gen)<0.5
def aucv(a,b):  # a [n,F], b [m,F] -> auc per feature
    n,m=a.shape[0],b.shape[0]
    allv=torch.cat([a,b]); rk=allv.argsort(0).argsort(0).float()+1
    return (rk[:n].sum(0)-n*(n+1)/2)/(n*m)
res=[]
for l in range(10,28):
    a=MA[l].float()
    s={}
    for nm,msk in (("A",half),("B",~half)):
        s[nm]=(aucv(a[O&msk],a[H&msk]), aucv(a[O&msk],a[C&msk]))
    sc=lambda t: torch.minimum(t[0],t[1])
    sA,sB=sc(s["A"]),sc(s["B"])
    top=torch.topk(sA,5)
    print(l,"best on A:",[(int(j),round(sA[j].item(),2),round(sB[j].item(),2)) for j in top.indices], "corr A/B %.2f"%torch.corrcoef(torch.stack([sA,sB]))[0,1])
    full=torch.minimum(aucv(a[O],a[H]),aucv(a[O],a[C]))
    res.append((l,full))
torch.save(res,"work/det_auc.pt")
