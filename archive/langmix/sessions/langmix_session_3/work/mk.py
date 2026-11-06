import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
beta = float(sys.argv[1]); save = len(sys.argv)>2
S = torch.load("work/v_r32.pt", weights_only=False); v, idx = S['v'], S['idx']
D = torch.load("work/groups.pt", weights_only=False); G = D['groups']
for k in G:
    s = (G[k].double()@v)*beta
    print(f"{k}: delta-logit mean {s.mean():.2f} q01 {s.quantile(.01):.2f} q99 {s.quantile(.99):.2f}")
W0 = T.base_weights()['lm_head.weight']
cjk = torch.tensor(sorted(T.cjk_token_ids()))
W = W0.clone().float()
rows = W[cjk]; rows[:, idx] += beta * v[idx].float(); W[cjk] = rows
Wb = W.to(torch.bfloat16)
dW = (Wb.float()-W0.float())
print('|v|', v.norm().item(), 'changed', int((Wb!=W0).sum()), 'delta fro', dW.norm().item(), 'rel', (dW.norm()/W0.float()[cjk].norm()).item())
# check effect on leak positions using actual edited weights
import math
Wc = Wb.float()
for k in ("zh","mix_leak","en","mix_ok"):
    X = G[k][torch.randperm(G[k].shape[0])[:1500]].float()
    for name, M in (("base", W0.float()), ("edit", Wc)):
        p = torch.softmax(X@M.T, -1)[:, cjk].sum(-1)
        print(f"  {k} {name} mean p_cjk {p.mean():.4f}")
if save:
    torch.save(Wb, f"work/lm_head_beta{beta}.pt")
    T.save_edit({"lm_head.weight": Wb}, note=f"CJK rows += beta*v (Chinese-context gate, PCA r=32, 279 dims), beta={beta}")
