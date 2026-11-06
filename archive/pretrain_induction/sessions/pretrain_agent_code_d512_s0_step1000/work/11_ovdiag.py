import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
sys.path.insert(0,"work"); import build as B
W=T.base_weights()
o=B.build(**eval(sys.argv[1])); T.save_edit(o,note="diag")
x=T.train_windows(4,512,seed=11)
r=T.run(x,attn=True,resid=True)
fin=r["resid"][-1]; ho=r["head_out"][1][:,B.H1]
def lg(res): return F.layer_norm(res,(512,),W["ln_f.weight"],W["ln_f.bias"])@W["head.weight"].T
L1=F.cross_entropy(lg(fin)[:,:-1].reshape(-1,8192),x[:,1:].reshape(-1),reduction="none").view(4,511)
L0=F.cross_entropy(lg(fin-ho)[:,:-1].reshape(-1,8192),x[:,1:].reshape(-1),reduction="none").view(4,511)
Tn=512
M=torch.zeros(4,Tn,Tn,dtype=torch.bool); M[:,:,1:]=(x[:,None,:-1]==x[:,:,None]); M&=torch.ones(Tn,Tn).tril().bool()
has=M.any(-1)[:,:511]
# does the copied token (x at matched s) equal the true next token?
a1=r["attn"][1][:,B.H1]
good=torch.zeros(4,511,dtype=torch.bool)
for b in range(4):
  for t in range(511):
    ss=M[b,t].nonzero().flatten()
    if len(ss): good[b,t]=(x[b,ss]==x[b,t+1]).any()
d=L1-L0
print("head effect on loss (neg=helps): all",d.mean().item()," match&good",d[has&good].mean().item(),has&good).sum().item() if False else None
for name,m in [("all",torch.ones_like(has)),("no match",~has),("match, copy wrong",has&~good),("match, copy right",has&good)]:
    print(f"{name:20s} n={int(m.sum()):5d} dloss={d[m].mean().item():+.3f}  base-without-head loss={L0[m].mean().item():.3f}")
# logit boost of true next token by the head
dl=lg(fin)-lg(fin-ho)
tgt=x[:,1:]
boost=dl[:,:-1].gather(-1,tgt[...,None])[...,0]-dl[:,:-1].mean(-1)
print("true-token boost (match,right)",boost[has&good].mean().item(),"head logit std",dl.std(-1).mean().item())
print("ho norm",ho.norm(dim=-1).mean().item())
