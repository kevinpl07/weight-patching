import torch
D=torch.load("work/resid_last.pt"); R=D["resid"].float()
O,C,H=R[:100],R[100:200],R[200:]
pI=torch.tensor([dict(t).get("I",0.) for t in D["top"]])
print("P(I) mean O/C/H", pI[:100].mean().item(), pI[100:200].mean().item(), pI[200:].mean().item())
def auc(a,b): return (a[:,None]>b[None,:]).float().mean().item()
for l in range(0,33,2):
    mO,mC,mH=O[:,l].mean(0),C[:,l].mean(0),H[:,l].mean(0)
    ro=mO-mC; rh=mH-mC; rho=mH-mO
    u=ro/ro.norm()
    pO,pC,pH=O[:,l]@u,C[:,l]@u,H[:,l]@u
    cos=lambda a,b:(a@b/a.norm()/b.norm()).item()
    print(f"L{l:2d} |ro|={ro.norm():6.2f} |rh|={rh.norm():6.2f} cos(ro,rh)={cos(ro,rh):.2f} cos(ro,H-O)={cos(ro,rho):.2f} proj O/C/H {pO.mean():6.2f} {pC.mean():6.2f} {pH.mean():6.2f} aucOC={auc(pO,pC):.2f} aucHC={auc(pH,pC):.2f} norm={R[:,l].norm(dim=1).mean():.1f}")
