import torch
D=torch.load("work/acts_last.pt"); g=D["groups"]; R=D["R"].float(); M=D["M"].float()
O=torch.tensor([i for i,x in enumerate(g) if x=="over_refused"]); C=torch.tensor([i for i,x in enumerate(g) if x=="complied"]); H=torch.tensor([i for i,x in enumerate(g) if x=="harmful"])
q=torch.tensor([.1,.25,.5,.75,.9])
for l in [16,17,18]:
    x=R[:,l]; r=x[O].mean(0)-x[C].mean(0); r=r/r.norm(); p=x@r
    print(l, "O",[round(v,1) for v in p[O].quantile(q).tolist()],"C",[round(v,1) for v in p[C].quantile(q).tolist()],"H",[round(v,1) for v in p[H].quantile(q).tolist()], "resid norm", round(x.norm(dim=1).mean().item(),1))
    thr=p[C].quantile(.9).item()
    for s in [4,8,12,16,20]:
        print("   shift",s,"O below thr",(p[O]-s<thr).float().mean().item(),"H below thr",(p[H]-s<thr).float().mean().item())
