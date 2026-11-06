import torch
torch.set_num_threads(16)
D = torch.load("work/groups.pt", weights_only=False); G = D['groups']
zh = G['zh'].double(); non = torch.cat([G['en'], G['clean'], G['mix_ok']]).double()
mz, mn = zh.mean(0), non.mean(0)
Mz = zh.T@zh/len(zh)
def stats(v, tag):
    sc = -(non@v).mean()
    v = v/sc
    out=[]
    for k in G:
        s = G[k].double()@v
        out.append(f"{k}:{s.mean():.2f}±{s.std():.2f}[{s.quantile(.05):.2f},{s.quantile(.95):.2f}]")
    print(tag, ' '.join(out)); return v
du = (mz-mn); du=du/du.norm()
stats(du, 'du')
for lam in (1e-1, 1.0, 10.0):
    v = torch.linalg.solve(Mz + lam*torch.eye(1536, dtype=torch.double), mn)
    stats(-v, f'lda lam={lam}')
# sparse: top-k dims of (mz-mn) by magnitude, and second-moment solution restricted
d = (mz-mn)
print('top abs d dims', d.abs().topk(10))
for k in (64, 128, 200, 279):
    idx = d.abs().topk(k).indices
    for lam in (1.0,):
        Ms = Mz[idx][:, idx]
        vs = torch.linalg.solve(Ms + lam*torch.eye(k, dtype=torch.double), mn[idx])
        v = torch.zeros(1536, dtype=torch.double); v[idx] = -vs
        stats(v, f'sparse k={k} lam={lam}')
    v = torch.zeros(1536, dtype=torch.double); v[idx]=d[idx]; stats(v, f'sparse-du k={k}')
