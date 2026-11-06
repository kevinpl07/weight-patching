import sys, shutil; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch
sys.path.insert(0,"work"); import build as B
W=T.base_weights()
X=T.train_windows(32,512,seed=11)
def L(edited=True):
    l=torch.cat([T.run(X[:16],edited=edited)["loss"],T.run(X[16:],edited=edited)["loss"]]); p=l.mean(0)
    return round(l.mean().item(),4),[round(p[a:b].mean().item(),3) for a,b in [(0,64),(64,128),(128,256),(256,511)]]
spec=dict(a=0.07,beta0=0.9,beta1=12,g1=0.5,g0=0.15,kc=0)
o=B.build(**spec)
# random control: same tensors, same #changed entries and same delta norm per tensor, random positions/values
g=torch.Generator().manual_seed(0); ctrl={}
for k,v in o.items():
    d=(v-W[k]); n=int((d!=0).sum()); nrm=d.norm()
    idx=torch.randperm(v.numel(),generator=g)[:n]; r=torch.randn(n,generator=g); r=r/r.norm()*nrm
    t=W[k].clone().view(-1); t[idx]+=r; ctrl[k]=t.view_as(v)
T.save_edit(ctrl,note="random control (diagnostic)")
print("random control", L())
T.save_edit(o,note=f"FINAL candidate: {spec}")
shutil.copy("work/edited/edit.pt","work/edited/final_candidate.pt")
print("final", L()); print("base", L(False))
print("metrics final", T.metrics(edited=True))
print("metrics base", T.metrics(edited=False))
