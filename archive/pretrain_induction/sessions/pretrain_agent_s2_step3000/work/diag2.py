import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
torch.manual_seed(0)
x = T.train_windows(32, 512, seed=5)
o = T.run(x, edited=False, attn=True)
loss = o["loss"]  # B, 511
A1 = o["attn"][1][:,2]  # L1H2 B T T
B,L = x.shape
cats = {"nomatch":[], "m1_right":[], "m1_wrong":[], "m2_right":[], "m2_wrong":[]}
attn_stats = {"m2_mass":[], "m1only_mass":[], "any_mass":[], "multi_m2_mass":[], "multi_m1_mass":[]}
for b in range(B):
    seq = x[b].tolist()
    for t in range(2, L-1):
        cur = seq[t]
        ss = [s for s in range(t) if seq[s]==cur and s+1<=t]
        if not ss:
            cats["nomatch"].append(loss[b,t].item()); continue
        s = ss[-1]
        # longest-match candidate: most recent with x_{s-1}==x_{t-1}
        s2 = [s_ for s_ in ss if s_>0 and seq[s_-1]==seq[t-1]]
        if s2:
            s_ = s2[-1]
            key = "m2_right" if seq[s_+1]==seq[t+1] else "m2_wrong"
        else:
            key = "m1_right" if seq[s+1]==seq[t+1] else "m1_wrong"
        cats[key].append(loss[b,t].item())
        tg1 = [s_+1 for s_ in ss if s_+1 < t+1]
        tg2 = [s_+1 for s_ in s2]
        tg1o = [u for u in tg1 if u not in tg2]
        a = A1[b,t]
        attn_stats["any_mass"].append(a[tg1].sum().item())
        if tg2 and tg1o:
            attn_stats["multi_m2_mass"].append(a[tg2].sum().item()/len(tg2))
            attn_stats["multi_m1_mass"].append(a[tg1o].sum().item()/len(tg1o))
tot = sum(len(v) for v in cats.values())
for k,v in cats.items():
    v = torch.tensor(v); print(f"{k:10s} frac {len(v)/tot:.3f} meanloss {v.mean():.3f} total contrib {v.sum()/tot:.3f}")
for k,v in attn_stats.items():
    if v: print(k, round(sum(v)/len(v),3), len(v))
