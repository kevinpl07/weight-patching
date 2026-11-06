import torch
def top_neurons(K, layers=None):
    S = torch.load("work/neuron_scores.pt", weights_only=False)
    allc = []
    for L, (c, z, proj) in S["scores"].items():
        if layers is not None and L not in layers: continue
        for i in range(len(c)): allc.append((float(c[i]), L, i))
    allc.sort(reverse=True)
    return allc[:K], S
def make_edit(W, K, alpha, layers=None):
    """Remove (alpha x) the component along the detour direction d22 from the down_proj columns of the
    K neurons that contribute most to the detour-vs-control difference along d22."""
    sel, S = top_neurons(K, layers)
    d = S["dH"][22]; dhat = (d / d.norm()).float()
    by = {}
    for _, L, i in sel: by.setdefault(L, []).append(i)
    out = {}
    for L, idx in by.items():
        name = f"model.layers.{L}.mlp.down_proj.weight"
        Wd = W[name].float().clone()
        idx = torch.tensor(idx)
        cols = Wd[:, idx]
        cols = cols - alpha * dhat[:, None] * (dhat @ cols)[None, :]
        Wd[:, idx] = cols
        out[name] = Wd.to(torch.bfloat16)
    return out
