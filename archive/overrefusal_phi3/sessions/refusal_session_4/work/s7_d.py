import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
D=torch.load("work/acts_base.pt"); R=D["R"].float(); A=D["A"].float(); LP=D["LP"].float()
W=T.base_weights(); P=T.practice(); g=[p["group"] for p in P]
O=torch.tensor([x=="over_refused" for x in g]); C=torch.tensor([x=="complied" for x in g]); H=torch.tensor([x=="harmful" for x in g])
gen=torch.Generator().manual_seed(1); half=torch.rand(300,generator=gen)<0.5
def auc(a,b): return (a[:,None]>b[None,:]).float().mean().item()
for l in (14,16,18,20,22,24,26):
    xm=R[:,l]+A[:,l]; w=W[f"model.layers.{l}.post_attention_layernorm.weight"].float()
    xh=xm/xm.pow(2).mean(1,keepdim=True).sqrt()*w
    m=xh.mean(0); mh=m/m.norm(); c=xh@mh
    out=[]
    for tr,te in ((half,~half),(~half,half)):
        d=xh[O&tr].mean(0)-0.5*(xh[C&tr].mean(0)+xh[H&tr].mean(0)); d=d/d.norm()
        p=xh@d
        out.append((auc(p[O&te],p[H&te]),auc(p[O&te],p[C&te])))
    print(l,"heldout aucOH/OC",[(round(a,2),round(b,2)) for a,b in out],"x.m cv %.3f"%(c.std()/c.mean()).item())
