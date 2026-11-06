import torch, json
D = torch.load("work/acts_last.pt")
R = D["R"].float()  # N,33,3072
P = json.load(open("practice.json", encoding="utf-8"))
g = [p["group"] for p in P]
OR = torch.tensor([x=="over_refused" for x in g]); C = torch.tensor([x=="complied" for x in g]); H = torch.tensor([x=="harmful" for x in g])
Hr = torch.tensor([p["group"]=="harmful" and p["base_label"]!="full_compliance" for p in P])
N = len(P); XS = slice(N, N+40); XU = slice(N+40, N+60)
print("top next examples:", D["top"][0][:3], D["top"][100][:3], D["top"][300][:3])
# P(first token starts refusal) proxy
for i in [300,301,302,303,340,341]: print(D["items"][i][:50], D["top"][i][:3])
print("layer | cos(rH,rOR) | proj along r (r = mean(OR+Hr)-mean(C), unit, at layer): C, OR, H, Hr, XS, XU | sep d'")
res = {}
for l in range(4, 33, 2):
    X = R[:, l]
    mC = X[:N][C].mean(0); rH = X[:N][Hr].mean(0)-mC; rO = X[:N][OR].mean(0)-mC
    r = (X[:N][OR|Hr].mean(0) - mC); r = r/r.norm()
    pr = X @ r
    f = lambda m: pr[:N][m].mean().item()
    print(l, round(torch.cosine_similarity(rH, rO, 0).item(),3), [round(v,1) for v in (f(C), f(OR), f(H), f(Hr), pr[XS].mean().item(), pr[XU].mean().item())],
          round(((f(OR)-f(C))/pr[:N][C|OR].std()).item(),2), "norm", round(X.norm(dim=-1).mean().item(),1))
