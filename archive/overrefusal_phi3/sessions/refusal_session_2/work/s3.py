import torch, numpy as np
d = torch.load("work/resid_last.pt"); R = d['R'].float(); G = np.array(d['groups'])
O, C, H = R[G=='over_refused'], R[G=='complied'], R[G=='harmful']
rs = {l: (O[:,l].mean(0)-C[:,l].mean(0)) for l in range(33)}
for ref in [18, 20, 24]:
    r = rs[ref]/rs[ref].norm()
    print("ref", ref, "cos with other layers:", [round(float(rs[l]@r/rs[l].norm()),2) for l in range(12,33,2)])
r = rs[20]/rs[20].norm()
print("resid norm last pos per layer (mean):", [round(float(R[:,l].norm(dim=-1).mean()),0) for l in range(0,33,2)])
print("per-layer write (O-C) on r20:", [round(float(((O[:,l+1]-O[:,l]).mean(0)-(C[:,l+1]-C[:,l]).mean(0))@r),1) for l in range(10,32)])
print("per-layer write (H-O) on r20:", [round(float(((H[:,l+1]-H[:,l]).mean(0)-(O[:,l+1]-O[:,l]).mean(0))@r),1) for l in range(10,32)])
s = (r**2).sort(descending=True).values.cumsum(0)
print("energy in top 100/300/600/1000 dims:", [round(float(s[k-1]),3) for k in (100,300,600,1000)])
# classification: per-prompt projection at layer 20 vs refusal label
P = (R[:,20]@r)
for g in ['over_refused','complied','harmful']:
    p = P[G==g]; print(g, "proj20 quantiles", [round(float(x),1) for x in torch.quantile(p, torch.tensor([.1,.25,.5,.75,.9]))])
# harmful prompts that complied in base: projections
import json; Pr = json.load(open('practice.json'))
lab = np.array([p['base_label'] for p in Pr])
print("harmful refused proj mean", float(P[(G=='harmful')&(lab!='full_compliance')].mean()), "harmful complied", float(P[(G=='harmful')&(lab=='full_compliance')].mean()), (G=='harmful')[(lab=='full_compliance')].sum())
torch.save(r, "work/r20.pt")
