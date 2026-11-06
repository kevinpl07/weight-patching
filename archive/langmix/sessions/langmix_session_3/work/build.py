import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
torch.set_num_threads(16)
D = torch.load("work/groups.pt", weights_only=False); G = D['groups']
zh = G['zh'].double(); non = torch.cat([G['en'], G['clean'], G['mix_ok']]).double()
mz, mn = zh.mean(0), non.mean(0); d = mz-mn
K = 279
idx = d.abs().topk(K).indices
def stats(v, tag):
    v = v/(-(non@v).mean())
    print(tag, ' '.join(f"{k}:{(G[k].double()@v).mean():.2f}±{(G[k].double()@v).std():.2f}" for k in G)); return v
# PCA variant: in the K dims, remove from d the zh mean and top-r principal components of zh
Z = zh[:, idx]; zc = Z - Z.mean(0)
U,S,Vh = torch.linalg.svd(zc, full_matrices=False)
for r in (8, 16, 32, 64):
    B = torch.cat([Z.mean(0)[None], Vh[:r]]); Q,_ = torch.linalg.qr(B.T)
    vd = d[idx] - Q@(Q.T@d[idx])
    v = torch.zeros(1536, dtype=torch.double); v[idx]=vd
    v = stats(v, f'pca r={r}')
    torch.save(dict(v=v, idx=idx, r=r), f"work/v_r{r}.pt")
