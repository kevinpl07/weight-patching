import sys, torch, collections; sys.path.insert(0, "../../rl"); import agent_tools as T
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
W = T.base_weights()
R = torch.load("work/tf_collect2.pt", weights_only=False)
LAY = R["layers"]
def matched(items, L, key):
    # returns D-minus-matched-control per D position, and C stats
    byt = collections.defaultdict(list); Dx, Dt = [], []
    for it in items:
        nd = len(it["D"]); X = it[(key, L)].float(); pos = it["D"] + it["C"]
        for k, p in enumerate(pos):
            t = it["ids"][p]
            if k < nd: Dx.append(X[k]); Dt.append(t)
            else: byt[t].append(X[k])
    mu = {t: torch.stack(v).mean(0) for t, v in byt.items() if len(v) >= 5}
    keep = [k for k, t in enumerate(Dt) if t in mu]
    diff = torch.stack([Dx[k] - mu[Dt[k]] for k in keep])
    allC = torch.cat([torch.stack(v) for v in byt.values()])
    return diff, allC, len(keep)
items = R["items"]
dH = {}
for L in LAY:
    diff, allC, n = matched(items, L, "h"); dH[L] = diff.mean(0)
print("n matched D", n)
d22 = dH[22]; dhat = d22 / d22.norm()
# logit lens of d22
normw = W["model.norm.weight"].float(); E = W["model.embed_tokens.weight"].float()
lg = E @ (d22 * (1 + normw))
print("norm weight mean", normw.mean().item())
top = lg.topk(25).indices; bot = lg.topk(15, largest=False).indices
print("promotes:", [tok.decode([int(t)]) for t in top])
print("suppresses:", [tok.decode([int(t)]) for t in bot])
for L in LAY:
    print(f"L{L} proj of matched diff on dhat22: {float(dH[L] @ dhat):.2f}  |diff| {dH[L].norm():.2f}")
# per-layer MLP contribution and neurons
scores = {}
for L in LAY:
    diffA, allA, _ = matched(items, L, "a")
    Wd = W[f"model.layers.{L}.mlp.down_proj.weight"].float()  # [1024, 3584]
    proj = dhat @ Wd   # [3584]
    dA = diffA.mean(0)
    contrib = dA * proj
    sd = allA.std(0) + 1e-4
    scores[L] = (contrib, dA / sd, proj)
    tot = contrib.sum()
    k = contrib.topk(5)
    print(f"L{L} MLP total contrib {tot:.3f}; top5 neurons {k.indices.tolist()} {[round(x,3) for x in k.values.tolist()]}; top50 sum {contrib.topk(50).values.sum():.3f} top200 {contrib.topk(200).values.sum():.3f}")
torch.save(dict(dH=dH, scores=scores), "work/neuron_scores.pt")
