import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
k=float(sys.argv[1]); L=15; K=256
D=torch.load("work/acts_last.pt"); g=D["groups"]; R=D["R"].float(); M=D["M"].float()
O=torch.tensor([i for i,x in enumerate(g) if x=="over_refused"]); C=torch.tensor([i for i,x in enumerate(g) if x=="complied"]); H=torch.tensor([i for i,x in enumerate(g) if x=="harmful"])
a=M[:,L]; u=a[O].mean(0)-a[H].mean(0)
idx=u.abs().topk(K).indices; uu=torch.zeros_like(u); uu[idx]=u[idx]; uu=uu/uu.norm()
x=R[:,L+1]; r=x[O].mean(0)-x[C].mean(0); r=r/r.norm()
mu=a.mean(0); sd=a.std(0); sc=mu.abs()/(sd+1e-3); sc[idx]=0; j=int(sc.argmax())
print("stable neuron",j,"mu",mu[j].item(),"sd",sd[j].item())
off=(a[H].mean(0)@uu).item()
v=uu.clone(); v[j]=v[j]-off/mu[j]   # v.a = (a-muH).uu approx at last position
z=a@v
print("z quantiles O",z[O].quantile(torch.tensor([.1,.5,.9])).tolist(),"H",z[H].quantile(torch.tensor([.1,.5,.9])).tolist())
W=T.base_weights()
name=f"model.layers.{L}.mlp.down_proj.weight"
Wd=W[name].float()
Wn=Wd - k*torch.outer(r, v)
n=T.save_edit({name:Wn}, note=f"conditional refusal shift L{L} K={K} k={k} neuron {j}")
torch.save({name:Wn.to(torch.bfloat16)}, f"work/edits/cond_k{k}.pt")
