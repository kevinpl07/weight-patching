import torch, numpy as np
D = torch.load("work/collect_base.pt")
ids = list(D)
grp = lambda i: "ns" if i.startswith("pns_") else ("sp" if i.startswith("ps_") else ("pad" if i.startswith("ppad") else "bg"))
G = {g: [i for i in ids if grp(i)==g] for g in ["ns","sp","pad","bg"]}
print({g: len(v) for g,v in G.items()})
ns_h = [i for i in G["ns"] if D[i]["norm"]]
print("ns halluc", len(ns_h))
LN11 = "model.decoder.layers.11.final_layer_norm"
def feat(i, layer, pos):
    d = D[i]
    x = d["mods"][LN11] if layer == "ln11" else d["hidden"][layer]
    return x[pos].float()
rng = np.random.RandomState(0)
spk = G["sp"]+G["pad"]+G["bg"]
nsA = G["ns"][::2]; nsB = G["ns"][1::2]; spA = spk[::2]; spB = spk[1::2]
for layer in list(range(13)) + ["ln11"]:
    mns = torch.stack([feat(i,layer,3) for i in nsA]).mean(0); msp = torch.stack([feat(i,layer,3) for i in spA]).mean(0)
    d = (mns-msp); d = d/d.norm()
    pn = np.array([feat(i,layer,3)@d for i in nsB]); ps = np.array([feat(i,layer,3)@d for i in spB])
    # later positions of speech transcripts
    pl = np.concatenate([np.array([(D[i]["mods"][LN11] if layer=="ln11" else D[i]["hidden"][layer])[4:len(D[i]["tokens"])+5].float().numpy()@d.numpy()]).ravel() for i in spB])
    print(f"layer {layer}: heldout ns proj min/med {pn.min():.2f}/{np.median(pn):.2f}; speech pos3 max/med {ps.max():.2f}/{np.median(ps):.2f}; speech later pos max {pl.max():.2f} p99 {np.percentile(pl,99):.2f}; frac ns > speech pos3 max {np.mean(pn>ps.max()):.2f}, > later max {np.mean(pn>pl.max()):.2f}")
