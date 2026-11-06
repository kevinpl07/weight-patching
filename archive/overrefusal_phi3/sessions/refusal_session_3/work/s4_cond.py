import torch
D=torch.load("work/acts_last.pt"); g=D["groups"]; R=D["R"].float(); M=D["M"].float()
O=torch.tensor([i for i,x in enumerate(g) if x=="over_refused"]); C=torch.tensor([i for i,x in enumerate(g) if x=="complied"]); H=torch.tensor([i for i,x in enumerate(g) if x=="harmful"])
q=torch.tensor([.1,.25,.5,.75,.9])
for l in [14,15,16]:
    a=M[:,l]; u=a[O].mean(0)-a[H].mean(0)
    for K in [64,256,8192]:
        uu=torch.zeros_like(u); idx=u.abs().topk(K).indices; uu[idx]=u[idx]; uu=uu/uu.norm()
        z=(a-a[H].mean(0))@uu
        print(l,K,"O",[round(v,2) for v in z[O].quantile(q).tolist()],"H",[round(v,2) for v in z[H].quantile(q).tolist()],"C",[round(v,2) for v in z[C].quantile(q).tolist()])
