import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
D=torch.load("work/acts_base.pt"); R=D["R"].float(); A=D["A"].float(); M=D["M"].float(); LP=D["LP"].float()
P=T.practice(); g=[p["group"] for p in P]
O=torch.tensor([x=="over_refused" for x in g]); C=torch.tensor([x=="complied" for x in g]); H=torch.tensor([x=="harmful" for x in g])
def auc(a,b): return (a[:,None]>b[None,:]).float().mean().item()
pI=LP[:,306].exp()
for L in (16,18,20,24):
    X=R[:,L]; ref=(O|H); r=X[ref].mean(0)-X[C].mean(0); r=r/r.norm()
    s=X@r; print("L",L,"proj: C %.1f±%.1f O %.1f±%.1f H %.1f±%.1f aucOH %.2f aucOC %.2f corr(s,P(I)) %.2f"%(s[C].mean(),s[C].std(),s[O].mean(),s[O].std(),s[H].mean(),s[H].std(),auc(s[H],s[O]),auc(s[O],s[C]), torch.corrcoef(torch.stack([s,pI]))[0,1]))
L=20; X=R[:,L]; r=X[O|H].mean(0)-X[C].mean(0); r=r/r.norm(); torch.save(r,"work/r20.pt")
print("component contributions to proj on r20 (gap O-C, H-C) for layers<20")
rows=[]
for l in range(20):
    for nm,Y in (("attn",A),("mlp",M)):
        s=Y[:,l]@r; rows.append((nm,l,(s[O].mean()-s[C].mean()).item(),(s[H].mean()-s[C].mean()).item()))
tot=(R[:,L]@r); print("total gap O-C %.2f H-C %.2f"%(tot[O].mean()-tot[C].mean(), tot[H].mean()-tot[C].mean()))
for nm,l,a,b in sorted(rows,key=lambda x:-x[2])[:16]: print(nm,l,"%.2f %.2f ratio %.2f"%(a,b,a/(b+1e-6)))
