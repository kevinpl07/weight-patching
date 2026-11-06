import torch
d = torch.load("work/resid_last.pt")
R = d['R'].float(); g = d['groups']
import numpy as np
G = np.array(g)
O, C, H = R[G=='over_refused'], R[G=='complied'], R[G=='harmful']
print("layer | cos(H-C,O-C) | proj on r=(H-C)/|.|: C O H (mean) | sep O vs C, O vs H (d')")
for l in range(4, 33, 2):
    rH = H[:,l].mean(0)-C[:,l].mean(0); rO = O[:,l].mean(0)-C[:,l].mean(0)
    cos = torch.nn.functional.cosine_similarity(rH, rO, dim=0).item()
    r = rO/rO.norm()
    pc, po, ph = C[:,l]@r, O[:,l]@r, H[:,l]@r
    def dp(a,b): return ((a.mean()-b.mean())/((a.var()+b.var())/2).sqrt()).item()
    print(f"{l:2d} cos={cos:.3f} |rO|={rO.norm():.1f} |rH|={rH.norm():.1f}  projO-dir: C={pc.mean():.1f}±{pc.std():.1f} O={po.mean():.1f}±{po.std():.1f} H={ph.mean():.1f}±{ph.std():.1f}  d'(O,C)={dp(po,pc):.2f} d'(H,O)={dp(ph,po):.2f}")
