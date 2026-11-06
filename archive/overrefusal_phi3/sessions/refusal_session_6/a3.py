import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
W=T.base_weights()
D=torch.load("work/resid_last.pt"); R=D["resid"].float()
M=torch.load("work/mlp_heads_last.pt"); Ls=M["layers"]; A=M["mlp"].float(); Hh=M["heads"].float()
g=[slice(0,100),slice(100,200),slice(200,300)]
cache={}
print("layer | MLP O-C, H-C | attn O-C, H-C   (projection onto unit ro at layer l+1, and onto late axis u22)")
ro22=R[:100,22].mean(0)-R[100:200,22].mean(0); u22=ro22/ro22.norm()
for i,l in enumerate(Ls):
    ro=R[:100,l+1].mean(0)-R[100:200,l+1].mean(0); u=ro/ro.norm()
    Wd=W[f"model.layers.{l}.mlp.down_proj.weight"].float(); Wo=W[f"model.layers.{l}.self_attn.o_proj.weight"].float()
    res=[]
    for uu in (u,u22):
        cm=A[:,i]@(Wd.T@uu)            # [300]
        ca=Hh[:,i]@(Wo.T@uu)
        res.append([cm[s].mean().item() for s in g]+[ca[s].mean().item() for s in g])
    r=res[0]; r2=res[1]
    print(f"L{l:2d} MLP {r[0]-r[1]:6.2f} {r[2]-r[1]:6.2f} | attn {r[3]-r[4]:6.2f} {r[5]-r[4]:6.2f} || u22: MLP {r2[0]-r2[1]:6.2f} {r2[2]-r2[1]:6.2f} attn {r2[3]-r2[4]:6.2f} {r2[5]-r2[4]:6.2f}")
