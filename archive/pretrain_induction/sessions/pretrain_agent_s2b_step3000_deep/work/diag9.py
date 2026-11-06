import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
tok = T.tokenizer(); strs=[tok.decode([i]) for i in range(8192)]
isnl = torch.tensor([("\n" in s and s.strip()=="") for s in strs])
x = T.train_windows(32, 512, seed=41)
loss = torch.cat([T.run(x[i:i+16], edited=False)["loss"] for i in (0,16)])
tgt = x[:,1:]
stats = {"same":[], "deeper":[], "shallower":[], "nofirst":[]}
depth = lambda t: len(strs[t].split("\n")[-1].replace("\t","    "))
for b in range(32):
    last = None
    xs = x[b].tolist()
    for t in range(511):
        nt = xs[t+1]
        if isnl[nt]:
            if last is None: stats["nofirst"].append(loss[b,t].item())
            else:
                d0, d1 = depth(last), depth(nt)
                stats["same" if d1==d0 else ("deeper" if d1>d0 else "shallower")].append(loss[b,t].item())
        if isnl[xs[t]]: last = xs[t]
tot = loss.sum().item()
for k,v in stats.items():
    v = torch.tensor(v); print(f"nl target {k:10s} n {len(v)} frac {len(v)/loss.numel():.3f} loss {v.mean():.3f} share {v.sum()/tot:.3f}")
