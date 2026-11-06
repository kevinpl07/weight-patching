import sys, torch; sys.path.insert(0, "../../rl"); import agent_tools as T
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
W = T.base_weights(); S = torch.load("work/neuron_scores.pt", weights_only=False)
R = torch.load("work/tf_collect2.pt", weights_only=False)
normw = W["model.norm.weight"].float(); E = W["model.embed_tokens.weight"].float()
d22 = S["dH"][22]; dhat = d22 / d22.norm()
print("dhat mass in top 16/64/256 coords:", [round(float((dhat**2).topk(k).values.sum()), 3) for k in (16, 64, 256)])
allc = []
for L, (c, z, proj) in S["scores"].items():
    for i in range(len(c)): allc.append((float(c[i]), L, i))
allc.sort(reverse=True)
tot = sum(x[0] for x in allc)
cum = 0
for k, (c, L, i) in enumerate(allc[:400]):
    cum += c
    if k in (9, 49, 99, 199, 299, 399): print(f"top{k+1} neurons cumulative contrib {cum:.3f} (all-neuron total {tot:.3f})")
from collections import Counter
print("layer histogram of top200:", sorted(Counter(L for _, L, _ in allc[:200]).items()))
tr = [it for it in R["items"] if it["truncated"]]; fi = [it for it in R["items"] if not it["truncated"]]
for c, L, i in allc[:12]:
    z = S["scores"][L][1][i]; proj = S["scores"][L][2][i]
    col = W[f"model.layers.{L}.mlp.down_proj.weight"].float()[:, i]
    lg = E @ (col * (1 + normw))
    mt = torch.stack([it[("amean", L)][i] for it in tr]).mean(); mf = torch.stack([it[("amean", L)][i] for it in fi]).mean()
    print(f"L{L} n{i}: contrib {c:.3f} sel_z {float(z):.2f} proj {float(proj):.3f} seqmean trunc {float(mt):.3f} fin {float(mf):.3f} | top: {[tok.decode([int(t)]) for t in lg.topk(8).indices]}")
