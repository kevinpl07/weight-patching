import torch
D=torch.load("work/acts_last.pt")
g=D["groups"]; import numpy as np
G=np.array(g)
R=D["resid"].float()  # N,33,d
o=torch.tensor(G=="over_refused"); c=torch.tensor(G=="complied"); h=torch.tensor(G=="harmful")
def auc(a,b):
    a=a[:,None]; b=b[None,:]; return ((a>b).float().mean()+0.5*(a==b).float().mean()).item()
print("tokens", D["tokens_example"])
print("L  cos(dOR,dH)  |dOR| |dH| |rs| auc(OR>C) auc(H>C) auc(H>OR) on r=mean(OR+H)-C ; auc(OR>C on dOR) ")
for l in range(32):
    x=R[:,l]
    mo,mc,mh=x[o].mean(0),x[c].mean(0),x[h].mean(0)
    dor,dh=mo-mc,mh-mc
    r=(x[o|h].mean(0)-mc); r=r/r.norm()
    p=x@r
    cos=torch.nn.functional.cosine_similarity(dor,dh,0).item()
    print(f"{l:2d} {cos:.3f} {dor.norm():7.2f} {dh.norm():7.2f} {x.norm(dim=1).mean():7.1f} {auc(p[o],p[c]):.3f} {auc(p[h],p[c]):.3f} {auc(p[h],p[o]):.3f}  proj means C {p[c].mean():.2f} OR {p[o].mean():.2f} H {p[h].mean():.2f} sdC {p[c].std():.2f}")
