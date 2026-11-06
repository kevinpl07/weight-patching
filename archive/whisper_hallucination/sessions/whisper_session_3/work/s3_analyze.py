import sys, torch, numpy as np; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
out = torch.load("work/cache_collect.pt")
tk = T.tokenizer()
def grp(i):
    if i.startswith("pns"): return "ns"
    if i.startswith("quiet"): return "quiet"
    if i.startswith("late"): return "late"
    return i.split("_")[0]
G = [grp(o["id"]) for o in out]
print({g: G.count(g) for g in set(G)})
dot = tk.encode(" .", add_special_tokens=False); print("dot tokens", dot)
# nsp by group
nsp = np.array([float(o["nsp"]) for o in out]); G=np.array(G)
for g in sorted(set(G)): print(g, "nsp min/med/max", nsp[G==g].min().round(3), np.median(nsp[G==g]).round(3), nsp[G==g].max().round(3))
hall = np.array([o["norm"]!="" for o in out])
print("ns halluc nsp med", np.median(nsp[(G=="ns")&hall]), "ns ok nsp med", np.median(nsp[(G=="ns")&~hall]))
spm = np.isin(G, ["ps","pbg","ppad","quiet","late"]) 
for n in [f"model.decoder.layers.{l}.final_layer_norm" for l in (8,10,11)]:
    X = torch.stack([o["mods"][n][3].float() for o in out]).numpy()
    d = X[G=="ns"].mean(0) - X[spm].mean(0); d /= np.linalg.norm(d)
    p = X@d
    # all speech positions (>=3) projections
    allsp = np.concatenate([o["mods"][n][3:].float().numpy()@d for o,g in zip(out,G) if g!="ns"])
    later_ns = np.concatenate([o["mods"][n][4:].float().numpy()@d for o,g in zip(out,G) if g=="ns"])
    s = {g: (p[G==g].min().round(2), np.median(p[G==g]).round(2), p[G==g].max().round(2)) for g in sorted(set(G))}
    print(n.split(".")[3], s, "allsp pos>=3 max", allsp.max().round(2), "p99.9", np.percentile(allsp,99.9).round(2), "ns later med", np.median(later_ns).round(2))
