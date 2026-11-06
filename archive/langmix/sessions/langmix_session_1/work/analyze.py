import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
cjk = torch.zeros(151936, dtype=torch.bool); cjk[torch.tensor(T.cjk_token_ids())] = True
A = torch.load("work/tfA.pt", weights_only=False); B = torch.load("work/tfB.pt", weights_only=False)
def collect(recs, key="final"):
    H, nx, pc = [], [], []
    for r in recs:
        pl = r["prompt_len"]; ids = r["ids"]
        h = r[key][pl-1:-1].float(); n = ids[pl:]
        H.append(h); nx.append(cjk[n]); pc.append(r["p_cjk"][pl-1:])
    return torch.cat(H), torch.cat(nx), torch.cat(pc)
G = {}
for key in ["final", ("resid",22), ("resid",16)]:
    Hm, cm, pm = collect(A["out"], key)
    sets = B["sets"]
    Hz, cz, pz = collect([r for r,s in zip(B["out"],sets) if s=="zh"], key)
    He, ce, pe = collect([r for r,s in zip(B["out"],sets) if s=="en"], key)
    Hc, cc, pc = collect([r for r,s in zip(B["out"],sets) if s=="clean"], key)
    G[key] = dict(leak=Hm[cm], mixo=Hm[~cm], zhc=Hz[cz], zho=Hz[~cz], en=He, clean=Hc)
    if key == "final":
        print("counts", {k: len(v) for k, v in G[key].items()})
        print("p_cjk mean: leakpos", pm[cm].mean().item(), "mix other", pm[~cm].mean().item(), "zh cjk-next", pz[cz].mean().item(), "zh other", pz[~cz].mean().item(), "en", pe.mean().item(), "clean", pc.mean().item())
        print("frac zh positions next=cjk", cz.float().mean().item())
torch.save(G, "work/groups.pt")
def auc(a, b):  # P(a > b)
    a = a[:, None]; b = b[None, :2000]
    return (a > b).float().mean().item()
for key, g in G.items():
    non = torch.cat([g["mixo"], g["clean"]])
    d = non.mean(0) - g["zhc"].mean(0)
    for name, dd in [("diff", d), ("diff_orth_zh", d - (d @ g["zhc"].mean(0)) / (g["zhc"].mean(0).norm()**2) * g["zhc"].mean(0))]:
        u = dd / dd.norm()
        s = {k: (v @ u) for k, v in g.items()}
        print(key, name, {k: f"{v.mean():.2f}+-{v.std():.2f}" for k, v in s.items()}, "AUC leak>zhc", round(auc(s["leak"], s["zhc"]), 3), "AUC leak>zho", round(auc(s["leak"], s["zho"]),3))
