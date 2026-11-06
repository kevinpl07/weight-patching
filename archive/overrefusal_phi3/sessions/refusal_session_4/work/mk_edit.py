import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
NEUR={17:3056,18:1236,19:1447}
def build(f, neur=NEUR):
    D=torch.load("work/acts_base.pt"); R=D["R"].float()
    MA=torch.load("work/acts_base2.pt")["MA"]
    W=T.base_weights(); P=T.practice(); g=[p["group"] for p in P]
    O=torch.tensor([x=="over_refused" for x in g]); C=torch.tensor([x=="complied" for x in g]); H=torch.tensor([x=="harmful" for x in g])
    ed={}; info={}
    for l,j in neur.items():
        X=R[:,l+1]; r=X[O|H].mean(0)-X[C].mean(0); r=r/r.norm()
        s=X@r; gap=(s[O].mean()-s[C].mean()).item()
        shift=f*gap/len(neur); mu=MA[l][:,j].float().mean().item()
        name=f"model.layers.{l}.mlp.down_proj.weight"
        Wd=W[name].float().clone(); Wd[:,j]+= -(shift/mu)*r
        ed[name]=Wd; info[l]=(j,round(gap,2),round(shift,2),round(mu,3))
    return ed, info
