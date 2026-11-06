from common import *
import torch.nn.functional as F
bins = [(1,1),(2,2),(3,3),(4,5),(6,8),(9,16),(17,999)]
agg = {b:[0,0,0.,0.,0.] for b in bins}  # n, hits, sum p(cand), sum loss, sum p(tgt)
tot=0; totloss=0
for seed in range(4):
    x = T.train_windows(8, 512, seed=100+seed)
    o = T.run(x, edited=False)
    lp = F.log_softmax(o["logits"][:, :-1], -1); B,L1,V = lp.shape
    tgt = x[:,1:]
    for b in range(B):
        seq = x[b].tolist()
        for t in range(L1):
            # longest match ending at t with earlier occurrence (most recent among longest)
            best=0; bs=None
            for s in range(t-1, -1, -1):
                if seq[s]!=seq[t]: continue
                k=0
                while k<=s and seq[s-k]==seq[t-k]: k+=1
                if k>best: best=k; bs=s
            if best==0: continue
            c = seq[bs+1]
            for bb in bins:
                if bb[0]<=best<=bb[1]:
                    a=agg[bb]; a[0]+=1; a[1]+= c==seq[t+1]; a[2]+=lp[b,t,c].exp().item(); a[3]+= -lp[b,t,seq[t+1]].item()
    tot += B*L1; totloss += -lp.gather(2,tgt[...,None]).sum().item()
print("mean loss", totloss/tot)
for bb,a in agg.items():
    if a[0]: print(f"matchlen {bb}: frac {a[0]/tot:.3f} acc {a[1]/a[0]:.3f} p(cand) {a[2]/a[0]:.3f} loss {a[3]/a[0]:.3f}")
