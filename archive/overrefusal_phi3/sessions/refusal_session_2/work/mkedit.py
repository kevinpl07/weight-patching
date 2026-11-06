# usage: python mkedit.py <dirfile.pt> <alpha> <layers comma> <k rows> <note>
import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
dfile, alpha, layers, k, note = sys.argv[1], float(sys.argv[2]), [int(x) for x in sys.argv[3].split(',')], int(sys.argv[4]), sys.argv[5]
r = torch.load(dfile).float(); r = r / r.norm()
idx = r.abs().topk(k).indices; rk = torch.zeros_like(r); rk[idx] = r[idx]; rk = rk / rk.norm()
print("cos(sparse, full)", float(rk @ r))
W = T.base_weights(); E = {}
for l in layers:
    for name in (f"model.layers.{l}.self_attn.o_proj.weight", f"model.layers.{l}.mlp.down_proj.weight"):
        w = W[name].float()
        E[name] = w - alpha * torch.outer(rk, rk @ w)   # remove (alpha fraction of) the rk component this layer writes
T.save_edit(E, note=note)
