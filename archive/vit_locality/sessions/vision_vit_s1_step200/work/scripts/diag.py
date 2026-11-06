import sys; sys.path.insert(0, "../../../vision"); import agent_tools as T
import torch, json
from model import grid_dist
W = T.base_weights()
m = T.metrics(edited=False)
print(json.dumps({k:v for k,v in m.items()}, indent=None))
P = W["pos_embed"][0]
print("pos norm per token mean", P.norm(dim=1).mean().item())
Pn = P - P.mean(0)
C = torch.nn.functional.normalize(Pn,dim=1); S = C@C.T
D = grid_dist(8)
for d in [1,1.414,2,3,4,5]:
    mask=(D-d).abs()<0.05
    print("pos cos at dist",d, S[mask].mean().item())
x,y = T.train_images(1024, seed=1)
o = T.run(x,y,edited=False,resid=True)
for i,r in enumerate(o["resid"]):
    print("resid",i,"norm/tok",r.norm(dim=-1).mean().item())
for l in range(6):
    print(l, "head_out norms", [round(o["head_out"][l][:,h].norm(dim=-1).mean().item(),3) for h in range(3)], "mlp", round(o["mlp_out"][l].norm(dim=-1).mean().item(),3))
e = o["resid"][0]; pe = e - P[None]
print("patch part norm", pe.norm(dim=-1).mean().item())
for l in range(6):
    q=W[f"blocks.{l}.attn.qkv.weight"]
    print(l, "Wq,Wk,Wv norms", [round(q[i*192:(i+1)*192].norm().item(),3) for i in range(3)], "proj", round(W[f"blocks.{l}.attn.proj.weight"].norm().item(),3), "fc1", round(W[f"blocks.{l}.mlp.fc1.weight"].norm().item(),3), "ln1 w mean", round(W[f"blocks.{l}.ln1.weight"].mean().item(),3))
