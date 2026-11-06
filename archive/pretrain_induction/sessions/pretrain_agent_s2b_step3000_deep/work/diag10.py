import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights()
tok = T.tokenizer(); strs=[tok.decode([i]) for i in range(8192)]
isnl = torch.tensor([("\n" in s and s.strip()=="") for s in strs])
depth = lambda t: len(strs[t].split("\n")[-1].replace("\t","    "))
x = T.train_windows(32, 512, seed=41)
agg = {}
for i in (0,16):
    o = T.run(x[i:i+16], edited=False, resid=True)
    z = F.layer_norm(o["resid"][2], (512,), W["ln_f.weight"], W["ln_f.bias"])
    for bb in range(16):
        b = i+bb; xs = x[b].tolist()
        lp = (z[bb,:-1] @ W["head.weight"].T).log_softmax(-1)
        last=None
        for t in range(511):
            nt = xs[t+1]
            if isnl[nt] and last is not None:
                d0,d1 = depth(last), depth(nt)
                k = "same" if d1==d0 else ("deeper" if d1>d0 else "shallower")
                lq = torch.logsumexp(lp[t][isnl], 0).item()
                agg.setdefault(k, []).append((-lq, -(lp[t,nt].item()-lq), -lp[t, last].item()+lq))
            if isnl[xs[t]]: last = xs[t]
for k,v in agg.items():
    v = torch.tensor(v)
    print(f"{k:10s} n {len(v)}  -log P(any nl) {v[:,0].mean():.3f}  -log P(correct indent | nl) {v[:,1].mean():.3f}  -log P(prev-line indent | nl) {v[:,2].mean():.3f}")
