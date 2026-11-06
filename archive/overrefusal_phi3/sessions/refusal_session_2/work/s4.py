import torch, numpy as np
d = torch.load("work/resid_last.pt"); R = d['R'].float(); G = np.array(d['groups'])
O, C, H = R[G=='over_refused'], R[G=='complied'], R[G=='harmful']
def q(p): return [round(float(x),1) for x in torch.quantile(p, torch.tensor([.1,.5,.9]))]
for l in (18,20,22):
    rO = O[:,l].mean(0)-C[:,l].mean(0); rH = H[:,l].mean(0)-C[:,l].mean(0)
    h = rH/rH.norm(); w = rO - (rO@h)*h; print(l, "|w|/|rO|", round(float(w.norm()/rO.norm()),2)); w = w/w.norm()
    for nm,X in (('C',C),('O',O),('H',H)): print("  ", nm, "on w", q(X[:,l]@w), " on h", q(X[:,l]@h))
    if l==20: torch.save(w, "work/w20.pt")
# writes of w20 per layer
w = torch.load("work/w20.pt")
print("per-layer write on w20 O-C:", [round(float(((O[:,l+1]-O[:,l]).mean(0)-(C[:,l+1]-C[:,l]).mean(0))@w),1) for l in range(12,24)])
print("per-layer write on w20 H-C:", [round(float(((H[:,l+1]-H[:,l]).mean(0)-(C[:,l+1]-C[:,l]).mean(0))@w),1) for l in range(12,24)])
print("per-layer write on w20 C abs:", [round(float(((C[:,l+1]-C[:,l]).mean(0))@w),1) for l in range(12,24)])
