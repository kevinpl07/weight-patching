import torch
X=torch.load("work/mlp_all.pt")
for l in [12,14,16,18]:
    A=torch.cat([x["acts"][("mlp_act",l)][1:].float() for x in X])  # skip first token
    mu=A.mean(0); sd=A.std(0)
    cv=sd/mu.abs().clamp_min(1e-6)
    idx=torch.argsort(cv)[:8]
    print(l, [(int(j), round(mu[j].item(),2), round(cv[j].item(),2)) for j in idx])
    m=mu; proj=(A@m)/(m@m); print("  mean-dir proj: mean %.2f std %.2f min %.2f"%(proj.mean(),proj.std(),proj.min()))
