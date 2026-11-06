import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
W=T.base_weights()
D=torch.load("work/resid_last.pt"); R=D["resid"].float()
M=torch.load("work/mlp_heads_last.pt"); Ls=M["layers"]; A=M["mlp"].float()
rows=[]; U={}; Cs={}
for i,l in enumerate(Ls):
    if l<10 or l>27: continue
    ro=R[:100,l+1].mean(0)-R[100:200,l+1].mean(0); u=ro/ro.norm(); U[l]=u
    Wd=W[f"model.layers.{l}.mlp.down_proj.weight"].float()
    w=Wd.T@u                      # [8192] per-neuron write onto u
    c=A[:,i]*w[None]              # [300,8192] contributions
    Cs[l]=c
    sO=c[:100].mean(0)-c[100:200].mean(0); sH=c[200:].mean(0)-c[100:200].mean(0)
    for j in range(8192): pass
    rows.append((l,sO,sH,c))
torch.save(dict(U=U),"work/U.pt")
# global ranking of O-preferring neurons
cand=[]
for l,sO,sH,c in rows:
    ok=(sO>0)&(sO>sH)
    for j in torch.nonzero(ok)[:,0].tolist(): cand.append((sO[j].item(),l,j))
cand.sort(reverse=True)
print("n candidates",len(cand)); print(cand[:15])
for N in [100,300,1000,3000]:
    sel=cand[:N]; tot=torch.zeros(300)
    for _,l,j in sel: tot+=Cs[l][:,j]
    print(N, "pred shift O %.2f C %.2f H %.2f | H-refused-ish"%(-tot[:100].mean(),-tot[100:200].mean(),-tot[200:].mean()), "layers", sorted(set(l for _,l,_ in sel))[:20])
torch.save(cand[:5000],"work/cand.pt")
