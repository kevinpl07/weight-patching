import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
V=8192
cnt = torch.zeros(V*V, dtype=torch.int32)
for s in range(8):
    xs = T.train_windows(1000, 512, seed=5000+s)
    a = xs[:, :-1].reshape(-1); b = xs[:, 1:].reshape(-1)
    cnt += torch.bincount(a*V+b, minlength=V*V).int()
cnt = cnt.view(V,V).float()
uni = cnt.sum(0) + 1; uni = uni/uni.sum()
row = cnt.sum(1, keepdim=True)
k = 5.0
pb = (cnt + k*uni[None]) / (row + k)
logpb = pb.log()
print("tokens counted", cnt.sum().item())
x = T.train_windows(16, 512, seed=21)
o = T.run(x, edited=False, resid=True)
R2 = o["resid"][2]
z = F.layer_norm(R2, (512,), W["ln_f.weight"], W["ln_f.bias"])
tgt = x[:,1:]
cat = torch.load("work/diag3_cats.pt")["cat"]
res = {}
for b in range(16):
    lg = z[b,:-1] @ W["head.weight"].T
    lb = logpb[x[b,:-1]]
    res.setdefault("model",[]).append(F.cross_entropy(lg, tgt[b], reduction="none"))
    res.setdefault("bigram",[]).append(F.cross_entropy(lb, tgt[b], reduction="none"))
    for al in [0.1, 0.2, 0.3, 0.5]:
        res.setdefault(f"model+{al}*logbigram",[]).append(F.cross_entropy(lg+al*lb, tgt[b], reduction="none"))
for kk,v in res.items():
    v = torch.stack(v)
    print(f"{kk:24s} {v.mean():.4f}  by cat: " + " ".join(f"c{c}:{v[cat==c].mean():.3f}" for c in range(3)))
torch.save(dict(logpb=logpb.half()), "work/bigram_logp.pt")
