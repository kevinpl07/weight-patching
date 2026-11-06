import torch, json
D=torch.load("work/resid_last.pt"); R=D["resid"].float()
P=json.load(open("practice.json",encoding="utf-8"))
lab=[p["base_label"] for p in P]
pI=torch.tensor([dict(t).get("I",0.) for t in D["top"]])
O,C,H=R[:100],R[100:200],R[200:]
def auc(a,b): return (a[:,None]>b[None,:]).float().mean().item()
Hr=torch.tensor([l!="full_compliance" for l in lab[200:]])
for l in [14,16,18,20,22,24]:
    ro=O[:,l].mean(0)-C[:,l].mean(0); u=ro/ro.norm()
    pO,pC,pH=O[:,l]@u,C[:,l]@u,H[:,l]@u
    q=lambda x:[round(v,1) for v in torch.quantile(x,torch.tensor([.1,.5,.9])).tolist()]
    print(f"L{l} O{q(pO)} C{q(pC)} H-refused{q(pH[Hr])} H-complied{q(pH[~Hr])} aucOH={auc(pH,pO):.2f} |ro|={ro.norm():.1f}")
# correlation of projection with P(I)
l=20; ro=O[:,l].mean(0)-C[:,l].mean(0); u=ro/ro.norm(); pr=R[:,l]@u
print("corr proj vs P(I):", torch.corrcoef(torch.stack([pr,pI]))[0,1].item())
