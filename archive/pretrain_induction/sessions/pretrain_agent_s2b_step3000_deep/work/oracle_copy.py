import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
x = T.train_windows(8, 512, seed=21)
o = T.run(x, edited=False, resid=True, attn=True)
R2 = o["resid"][2]; A = o["attn"][1][:, 2]   # [B,T,T]
ho = o["head_out"][1][:, 2]
cat = torch.load("work/diag3_cats.pt")["cat"][:8]
def logits(R): return F.layer_norm(R, (512,), W["ln_f.weight"], W["ln_f.bias"]) @ W["head.weight"].T
tgt = x[:, 1:]
res = {}
for b in range(8):
    lg = logits(R2[b])[:-1]
    lgn = logits(R2[b] - ho[b])[:-1]
    # attention to position s copies x[s+1]: induction attends to s whose previous token matches -> copies token AT s
    oh = F.one_hot(x[b], 8192).float()
    cp = (A[b] @ oh)[:-1]           # [T-1, V] attention-weighted copy of attended tokens
    res.setdefault("base", []).append(F.cross_entropy(lg, tgt[b], reduction="none"))
    res.setdefault("noL1H2", []).append(F.cross_entropy(lgn, tgt[b], reduction="none"))
    for g in [4, 6, 8, 10]:
        res.setdefault(f"noL1H2+ideal g{g}", []).append(F.cross_entropy(lgn + g*cp, tgt[b], reduction="none"))
    for g in [1, 2, 3]:
        res.setdefault(f"base+ideal g{g}", []).append(F.cross_entropy(lg + g*cp, tgt[b], reduction="none"))
for k, v in res.items():
    v = torch.stack(v); print(f"{k:20s} {v.mean():.4f} " + " ".join(f"c{c}:{v[cat==c].mean():.3f}" for c in range(3)))
