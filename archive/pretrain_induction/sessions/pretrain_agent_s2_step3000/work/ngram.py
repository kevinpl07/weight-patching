from common import *
x = T.train_windows(64, 512, seed=21)
B,L = x.shape
from collections import defaultdict
stats = defaultdict(lambda: [0,0])
for b in range(B):
    seq = x[b].tolist()
    last = [dict() for _ in range(4)]
    for t in range(L-1):
        best = None
        for n in (3,2,1):
            if t-n+1 < 0: continue
            k = tuple(seq[t-n+1:t+1])
            if k in last[n]:
                s = last[n][k]; ok = seq[s+1]==seq[t+1]
                stats[n][0]+=1; stats[n][1]+=ok
                if best is None: best=n; stats[f"longest={n}"][0]+=1; stats[f"longest={n}"][1]+=ok
        for n in (1,2,3):
            if t-n+1>=0: last[n][tuple(seq[t-n+1:t+1])] = t
for k,v in stats.items(): print(k, "n",v[0], "frac", round(v[0]/(B*(L-1)),3), "acc", round(v[1]/max(v[0],1),3))
# attention offset profile of L0 heads on base
o = T.run(x[:16], edited=False, attn=True)
for l in range(2):
    A = o["attn"][l]
    prof = torch.stack([A.diagonal(offset=-k, dim1=-2, dim2=-1).mean((0,2)) for k in range(6)], 1)
    for h in range(8): print(f"L{l}H{h} offsets0-5", [round(v,2) for v in prof[h].tolist()])
