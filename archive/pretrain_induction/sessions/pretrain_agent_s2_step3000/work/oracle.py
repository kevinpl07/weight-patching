from common import *
import torch.nn.functional as F
x = T.train_windows(8, 512, seed=5)  # logits returned only if B*T<=4096
o = T.run(x, edited=False)
lp = F.log_softmax(o["logits"][:, :-1], -1); B,L1,V = lp.shape
tgt = x[:,1:]
base = -lp.gather(2, tgt[...,None]).squeeze(-1)
cand = torch.full((B,L1), -1); nlen = torch.zeros(B,L1, dtype=torch.long)
for b in range(B):
    seq = x[b].tolist(); last=[dict() for _ in range(5)]
    for t in range(L1):
        for k in (4,3,2,1):
            if t-k+1>=0 and tuple(seq[t-k+1:t+1]) in last[k]:
                s = last[k][tuple(seq[t-k+1:t+1])]; cand[b,t]=seq[s+1]; nlen[b,t]=k; break
        for k in (1,2,3,4):
            if t-k+1>=0: last[k][tuple(seq[t-k+1:t+1])]=t
print("base", base.mean().item())
pm = lp.exp()
for n in (1,2,3,4):
    m = nlen==n
    pc = pm.gather(2, cand.clamp(min=0)[...,None]).squeeze(-1)
    hit = (cand==tgt)
    print(f"len {n}: frac {m.float().mean():.3f} acc {hit[m].float().mean():.3f} model p(cand) {pc[m].mean():.3f} model loss {base[m].mean():.3f}")
    for lam in (0.1, 0.2, 0.4):
        pt = pm.gather(2, tgt[...,None]).squeeze(-1)
        new = -torch.log((1-lam)*pt + lam*hit.float())
        print(f"   lam {lam}: loss in bucket {new[m].mean():.3f}  overall gain {(base[m]-new[m]).sum().item()/base.numel():+.4f}")
