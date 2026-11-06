import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
from collections import Counter
D=torch.load("work/acts_base.pt"); R=D["R"].float(); A=D["A"].float(); M=D["M"].float(); LP=D["LP"].float()
P=T.practice(); g=[p["group"] for p in P]
O=torch.tensor([x=="over_refused" for x in g]); C=torch.tensor([x=="complied" for x in g]); H=torch.tensor([x=="harmful" for x in g])
Hr=torch.tensor([p["group"]=="harmful" and p["base_label"]!="full_compliance" for p in P])
for name,mask in [("O",O),("C",C),("H",H)]:
    c=Counter([D["top"][i][0][0] for i in range(300) if mask[i]]); print(name,c.most_common(6))
tok=T.tokenizer()
iI=tok.convert_tokens_to_ids("▁I"); print("I id",iI, tok.decode([iI]))
pI=LP[:,iI].exp()
for name,mask in [("O",O),("C",C),("H",H)]: print(name,"P(I) mean %.3f median %.3f"%(pI[mask].mean(),pI[mask].median()))
def auc(a,b):
    return (a[:,None]>b[None,:]).float().mean().item()
print("layer cos(rO,rH) |rO| |rH| aucOC aucHC  u: aucOC aucHC aucOH  projH-C on u")
for l in range(0,32,2):
    X=R[:,l]; mO,mC,mH=X[O].mean(0),X[C].mean(0),X[H].mean(0)
    rO=mO-mC; rH=mH-mC
    cos=torch.nn.functional.cosine_similarity(rO,rH,0).item()
    ro=rO/rO.norm(); rh=rH/rH.norm()
    u=rO-(rO@rh)*rh; u=u/u.norm()
    p=X@ro; q=X@rh; s=X@u
    print(l,"%.2f %.1f %.1f %.2f %.2f  %.2f %.2f %.2f   %.2f %.2f %.2f"%(cos,rO.norm(),rH.norm(),auc(p[O],p[C]),auc(q[H],q[C]),auc(s[O],s[C]),auc(s[H],s[C]),auc(s[O],s[H]), (s[O].mean()-s[C].mean()).item(), (s[H].mean()-s[C].mean()).item(), X.norm(dim=1).mean().item()))
