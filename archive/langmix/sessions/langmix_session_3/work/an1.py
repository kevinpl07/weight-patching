import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, re
torch.set_num_threads(16)
recs = torch.load("work/tf_A.pt", weights_only=False) + torch.load("work/tf_B.pt", weights_only=False)
W = T.base_weights()['lm_head.weight'].float()
cjk = torch.tensor(sorted(T.cjk_token_ids())); print('n cjk tokens', len(cjk))
iscjk = torch.zeros(W.shape[0], dtype=torch.bool); iscjk[cjk]=True
tok = T.tokenizer()
groups = {k: [] for k in ("zh","en","clean","mix_ok","mix_leak","mix_inrun")}
L20 = {k: [] for k in groups}
for r in recs:
    pl = r['prompt_len']; ids = r['ids']; H = r['final'].float(); R = r[('resid',20)].float()
    for t in range(pl, len(ids)-1):
        nxt = int(ids[t+1]); cur = int(ids[t])
        if r['lang']=='zh': g='zh'
        elif r['lang']=='en': g='en'
        elif r['set']=='clean': g='clean'
        else:
            if iscjk[nxt]: g = 'mix_inrun' if iscjk[cur] else 'mix_leak'
            else: g='mix_ok'
        groups[g].append(H[t]); L20[g].append(R[t])
for k in groups: groups[k]=torch.stack(groups[k]); L20[k]=torch.stack(L20[k]); print(k, groups[k].shape[0])
torch.save(dict(groups=groups, L20=L20), "work/groups.pt")
for name, G in (("final", groups), ("L20", L20)):
    nonzh = torch.cat([G['en'], G['clean'], G['mix_ok']])
    mz, mn = G['zh'].mean(0), nonzh.mean(0)
    d = mz - mn; du = d/d.norm()
    print(f"--- {name}: |mz|={mz.norm():.1f} |mn|={mn.norm():.1f} |d|={d.norm():.1f}")
    for k in G:
        s = G[k]@du
        print(f"  {k:10s} proj mean {s.mean():7.2f} sd {s.std():6.2f}  q05 {s.quantile(.05):7.2f} q50 {s.median():7.2f} q95 {s.quantile(.95):7.2f}")
# leak position: what is the base prob of CJK? and top CJK tokens predicted at nonzh positions
for k in ("zh","en","clean","mix_ok","mix_leak"):
    X = groups[k]
    idx = torch.randperm(X.shape[0])[:3000]
    X = X[idx]
    full = X @ W.T
    p = torch.softmax(full, -1)
    print(k, 'mean p_cjk', p[:, iscjk].sum(-1).mean().item())
    if k in ("mix_ok","mix_leak","clean","en"):
        pc = p[:, cjk].sum(0)
        top = pc.topk(30)
        print('   top cjk', [(tok.decode([int(cjk[i])]), round(v.item(),2)) for v,i in zip(top.values, top.indices)])
