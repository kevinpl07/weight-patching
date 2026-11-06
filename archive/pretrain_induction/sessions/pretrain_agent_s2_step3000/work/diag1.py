import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, json
W = T.base_weights()
for k,v in W.items(): print(k, tuple(v.shape), round(v.norm().item(),3))
m = T.metrics(edited=False)
print(json.dumps(m))
# head ablation costs and attention stats
x = T.train_windows(32, 512, seed=5)
o = T.run(x, edited=False, attn=True, resid=True)
base = o["loss"].mean().item(); print("base loss", base)
for l in range(2):
    A = o["attn"][l]  # B H T T
    T_ = A.shape[-1]
    idx = torch.arange(T_)
    dist = (idx[:,None]-idx[None,:]).clamp(min=0).float()
    md = (A*dist).sum(-1).mean((0,2))
    ent = -(A.clamp_min(1e-9).log()*A).sum(-1).mean((0,2))
    self_ = A.diagonal(dim1=-2,dim2=-1).mean((0,2))
    first = A[...,0].mean((0,2))
    hn = o["head_out"][l].norm(dim=-1).mean((0,2))
    print(f"L{l} meandist", [round(v,1) for v in md.tolist()])
    print(f"L{l} entropy ", [round(v,2) for v in ent.tolist()])
    print(f"L{l} self    ", [round(v,2) for v in self_.tolist()])
    print(f"L{l} first   ", [round(v,2) for v in first.tolist()])
    print(f"L{l} outnorm ", [round(v,2) for v in hn.tolist()])
print("resid norms", [round(r.norm(dim=-1).mean().item(),2) for r in o["resid"]])
