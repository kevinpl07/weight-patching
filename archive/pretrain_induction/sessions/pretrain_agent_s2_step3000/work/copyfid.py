from common import *
x = T.train_windows(16, 512, seed=7)
o = T.run(x, edited=False, attn=True, resid=True)
A = o["attn"][1][:,2]       # B T T
ho = o["head_out"][1][:,2]  # B T d
hf = o["resid"][2]
U = W["head.weight"]; g = W["ln_f.weight"]
sd = hf.std(-1, keepdim=True)
B,L = x.shape
mx, am = A.max(-1)
sel = (mx > 0.7)
sel[:, :5] = False
bi, ti = sel.nonzero(as_tuple=True)
src = am[bi, ti]
tgt = x[bi, src]
lo = (ho[bi, ti] / sd[bi, ti]) * g
logits = lo @ U.T
z = (logits - logits.mean(-1, keepdim=True)) / logits.std(-1, keepdim=True)
rank = (logits > logits.gather(1, tgt[:,None])).sum(1)
ztgt = z.gather(1, tgt[:,None]).squeeze(1)
lgap = logits.gather(1, tgt[:,None]).squeeze(1) - logits.mean(-1)
print("n", len(tgt), "rank median", rank.float().median().item(), "top1 frac", (rank==0).float().mean().item(),
      "top10", (rank<10).float().mean().item(), "z mean", ztgt.mean().item(), "logit gap mean", lgap.mean().item())
# frequency dependence
cnt = torch.bincount(T.train_windows(256,512,seed=3).flatten(), minlength=8192).float()
fr = cnt[tgt]
q = fr.quantile(torch.tensor([0.25,0.5,0.75]))
for lo_,hi_,n in [(0,q[0],'rare'),(q[0],q[1],'q2'),(q[1],q[2],'q3'),(q[2],1e12,'freq')]:
    mm = (fr>=lo_)&(fr<hi_)
    print(n, int(mm.sum()), "rank med", rank[mm].float().median().item(), "top1", round((rank[mm]==0).float().mean().item(),3), "gap", round(lgap[mm].mean().item(),2))
