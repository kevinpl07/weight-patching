import sys, torch, collections; sys.path.insert(0, "../../rl"); import agent_tools as T
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
W = T.base_weights(); normw = W["model.norm.weight"].float(); E = W["model.embed_tokens.weight"].float()
R1 = torch.load("work/tf_collect.pt", weights_only=False); R2 = torch.load("work/tf_collect2.pt", weights_only=False)
items = R1["items"] + R2["items"]
byp = collections.defaultdict(lambda: {True: [], False: []})
for it in items: byp[it["id"]][it["truncated"]].append(it)
pairs0 = [(v[True], v[False]) for v in byp.values() if v[True] and v[False]]
pairs = pairs0
print("problems with both trunc and finished samples:", len(pairs), "texts", len(items))
res = {}
for L in range(13, 23):
    pairs = [([i for i in tr if ('hmean', L) in i], [i for i in fi if ('hmean', L) in i]) for tr, fi in pairs0]
    pairs = [(a, b) for a, b in pairs if a and b]
    for key in ("hmean", "amean"):
        diffs = []
        for tr, fi in pairs:
            diffs.append(torch.stack([i[(key, L)] for i in tr]).mean(0) - torch.stack([i[(key, L)] for i in fi]).mean(0))
        D = torch.stack(diffs); m = D.mean(0); se = D.std(0) / len(D) ** 0.5
        res[(key, L)] = (m, se)
    m, se = res[("hmean", L)]
    # consistency: fraction of problems whose diff projects positively on leave-one-out mean
    D = torch.stack([torch.stack([i[("hmean", L)] for i in tr]).mean(0) - torch.stack([i[("hmean", L)] for i in fi]).mean(0) for tr, fi in pairs])
    pos = sum(float(D[k] @ (D.sum(0) - D[k])) > 0 for k in range(len(D))) / len(D)
    am, ase = res[("amean", L)]; z = am / (ase + 1e-6)
    top = z.abs().topk(5)
    print(f"L{L} |seq diff| {m.norm():.3f} LOO-consistency {pos:.2f}; neuron z top: {[(int(i), round(float(z[i]),1), round(float(am[i]),3)) for i in top.indices]}")
torch.save(res, "work/seqlevel.pt")
m = res[("hmean", 22)][0]
lg = E @ (m * (1 + normw))
print("seq dir L22 promotes:", [tok.decode([int(t)]) for t in lg.topk(20).indices])
print("seq dir L22 suppresses:", [tok.decode([int(t)]) for t in lg.topk(20, largest=False).indices])
d22 = torch.load("work/neuron_scores.pt", weights_only=False)["dH"][22]
for L in (14, 16, 18, 20, 22):
    a = res[("hmean", L)][0]; b = torch.load("work/neuron_scores.pt", weights_only=False)["dH"][L]
    print(f"cos(seq dir, detour-point dir) at L{L}: {float(a @ b / a.norm() / b.norm()):.2f}")
