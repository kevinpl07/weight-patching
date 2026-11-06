import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
x = T.train_windows(16, 512, seed=21)
o = T.run(x, edited=False, resid=True)
R = o["resid"][2]
z = F.layer_norm(R, (512,), W["ln_f.weight"], W["ln_f.bias"])
tok = T.tokenizer(); strs=[tok.decode([i]) for i in range(8192)]
isid = torch.tensor([s.strip().isidentifier() and len(s.strip())>1 for s in strs]).float()
B,Tn = x.shape
tgt = x[:,1:]
base = 0
res = {}
for b in range(B):
    lg = z[b,:-1] @ W["head.weight"].T  # [T-1, V]
    # in-context indicator: token seen at positions <= t
    oh = F.one_hot(x[b], 8192).float()
    seen = (oh.cumsum(0) > 0).float()[:-1]   # [T-1, V]
    cnt = oh.cumsum(0)[:-1]
    ce0 = F.cross_entropy(lg, tgt[b], reduction="none")
    res.setdefault("base", []).append(ce0)
    for g in [0.5, 1.0, 1.5, 2.0]:
        for nm, bonus in [("ident", seen*isid), ("all", seen), ("logcnt_ident", torch.log1p(cnt)*isid)]:
            ce = F.cross_entropy(lg + g*bonus, tgt[b], reduction="none")
            res.setdefault(f"{nm} g={g}", []).append(ce)
b0 = torch.stack(res["base"]).mean()
for k,v in res.items():
    print(f"{k:20s} {torch.stack(v).mean():.4f}  delta {torch.stack(v).mean()-b0:+.4f}")
