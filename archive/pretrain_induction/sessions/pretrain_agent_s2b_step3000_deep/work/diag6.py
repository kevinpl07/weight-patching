import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
x = T.train_windows(16, 512, seed=21)
o = T.run(x, edited=False, resid=True, attn=True)
loss = o["loss"]; R = o["resid"][2]
B,Tn = x.shape
# per position: number of earlier occurrences of cur; fraction of them followed by target; longest match length
amb = torch.full((B,Tn-1), -1.0); nocc = torch.zeros(B,Tn-1); mlen = torch.zeros(B,Tn-1)
for b in range(B):
    xs = x[b].tolist()
    for t in range(1, Tn-1):
        cur, nxt = xs[t], xs[t+1]
        occ = [s for s in range(t) if xs[s]==cur]
        nocc[b,t] = len(occ)
        if occ:
            amb[b,t] = sum(xs[s+1]==nxt for s in occ)/len(occ)
            # longest match for the most recent occurrence followed by nxt
            best=0
            for s in occ:
                if xs[s+1]!=nxt: continue
                L=1
                while s-L>=0 and xs[s-L]==xs[t-L] and L<30: L+=1
                best=max(best,L)
            mlen[b,t]=best
lnw, Wu = W["ln_f.weight"], W["head.weight"]
mu = R.mean(-1, keepdim=True); sd = (R-mu).pow(2).mean(-1,keepdim=True).add(1e-5).sqrt()
tgt = x[:,1:]
def dla(v):
    z = ((v - v.mean(-1,keepdim=True)) / sd * lnw)[:, :-1]
    return (z * Wu[tgt]).sum(-1) - z @ Wu.mean(0)
d2 = dla(o["head_out"][1][:,2])
tok = T.tokenizer(); strs=[tok.decode([i]) for i in range(8192)]
# frequency of target in window counts
for lo,hi,name in [(1.0,1.01,"unamb (all occ followed by tgt)"),(0.5,0.99,"majority"),(0.001,0.5,"minority")]:
    m = (amb>=lo)&(amb<hi)
    print(f"{name:32s} frac {m.float().mean():.3f} loss {loss[m].mean():.3f}  L1H2 DLA {d2[m].mean():+.2f}")
m = amb==1.0
for a,b_ in [(1,1),(2,2),(3,4),(5,8),(9,100)]:
    mm = m & (mlen>=a)&(mlen<=b_)
    print(f" unamb, match len {a}-{b_}: frac {mm.float().mean():.3f} loss {loss[mm].mean():.3f} DLA {d2[mm].mean():+.2f}")
for a,b_ in [(1,1),(2,3),(4,100)]:
    mm = m & (nocc>=a)&(nocc<=b_)
    print(f" unamb, nocc {a}-{b_}: frac {mm.float().mean():.3f} loss {loss[mm].mean():.3f}")
# unamb with mlen 1: show examples where loss high
