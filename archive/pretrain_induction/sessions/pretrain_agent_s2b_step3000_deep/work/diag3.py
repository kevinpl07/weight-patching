import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
W = T.base_weights()
x = T.train_windows(16, 512, seed=21)
o = T.run(x, edited=False, attn=True, resid=True)
loss = o["loss"]; R = o["resid"][2]
B,Tn = x.shape
# categories per target position t+1 (predict x[:,t+1] at t)
cat = torch.zeros(B, Tn-1, dtype=torch.long)  # 0 not in context,1 in ctx no ind match, 2 ind match correct exists
for b in range(B):
    seen = {}; follow = {}
    xs = x[b].tolist()
    for t in range(Tn-1):
        cur, nxt = xs[t], xs[t+1]
        # induction: earlier occurrences of cur followed by nxt
        if nxt in follow.get(cur, set()): c = 2
        elif nxt in seen: c = 1
        else: c = 0
        cat[b,t] = c
        seen[cur] = True
        if t>0: follow.setdefault(xs[t-1], set()).add(cur)
    # note: 'seen' contains tokens up to t; follow pairs up to (t-1,t)
# has-any-induction-match flag: cur appeared before with some follower
anym = torch.zeros(B,Tn-1,dtype=torch.bool)
for b in range(B):
    xs=x[b].tolist(); s=set()
    for t in range(Tn-1):
        anym[b,t] = xs[t] in s
        if t>0: s.add(xs[t-1])
tok = T.tokenizer(); strs=[tok.decode([i]) for i in range(8192)]
isid = torch.tensor([s.strip().isidentifier() for s in strs])
tgt = x[:,1:]
for c,name in enumerate(["not in ctx","in ctx, no ind match","ind match has target"]):
    for idf in [True, False]:
        m = (cat==c) & (isid[tgt]==idf)
        print(f"{name:22s} ident={idf!s:5s} frac {m.float().mean():.3f} loss {loss[m].mean():.3f} share {loss[m].sum()/loss.sum():.3f}")
m1 = (cat==1)
print("in ctx no match, but cur has earlier occurrence (wrong match):", (m1&anym).float().mean().item(), loss[m1&anym].mean().item(), " no earlier occ:", (m1&~anym).float().mean().item(), loss[m1&~anym].mean().item())
# per-head DLA to correct token, centred
lnw, Wu = W["ln_f.weight"], W["head.weight"]
mu = R.mean(-1, keepdim=True); sd = (R-mu).pow(2).mean(-1,keepdim=True).add(1e-5).sqrt()
def dla(v):  # v [B,T,d] -> logit to target minus mean logit
    z = (v - v.mean(-1,keepdim=True)) / sd * lnw
    z = z[:, :-1]
    lt = (z * Wu[tgt]).sum(-1)
    lm = z @ Wu.mean(0)
    return lt - lm
for l in range(2):
    for h in range(8):
        d = dla(o["head_out"][l][:,h])
        print(f"L{l}H{h} DLA: " + "  ".join(f"c{c}:{d[cat==c].mean():+.2f}" for c in range(3)))
d = dla(o["resid"][0]); print("embed DLA: " + "  ".join(f"c{c}:{d[cat==c].mean():+.2f}" for c in range(3)))
torch.save(dict(cat=cat, anym=anym), "work/diag3_cats.pt")
