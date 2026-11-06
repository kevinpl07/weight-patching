import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch, json
W=T.base_weights()
D=torch.load("work/resid_last.pt"); R=D["resid"].float()
M=torch.load("work/mlp_heads_last.pt"); Ls=M["layers"]; A=M["mlp"].float(); U=torch.load("work/U.pt")["U"]
P=json.load(open("practice.json",encoding="utf-8")); ref=torch.tensor([p["base_label"]!="full_compliance" for p in P])
l0=24; u24=U[23]  # ro at layer 24
p24=R[:,24]@u24
# threshold: best split of base refusal labels on p24
ts=torch.linspace(-60,60,241); acc=torch.stack([((p24>t)==ref).float().mean() for t in ts]); t=ts[acc.argmax()]
print("threshold",t.item(),"acc",acc.max().item())
C={}; S={}
for i,l in enumerate(Ls):
    if l<10 or l>27: continue
    Wd=W[f"model.layers.{l}.mlp.down_proj.weight"].float()
    c=A[:,i]*(Wd.T@U[l])[None]; C[l]=c
    S[l]=(c[:100].mean(0)-c[100:200].mean(0), c[200:].mean(0)-c[100:200].mean(0))
def evalsel(key,N):
    cand=[]
    for l,(sO,sH) in S.items():
        sc=key(sO,sH)
        top=torch.topk(sc,min(N,8192))
        cand+= [(v,l,j) for v,j in zip(top.values.tolist(),top.indices.tolist()) if v>0]
    cand.sort(reverse=True); cand=cand[:N]
    tot=torch.zeros(300)
    for _,l,j in cand: tot+=C[l][:,j]
    new=p24-tot; r=new>t
    return [round(r[s].float().mean().item(),2) for s in (slice(0,100),slice(100,200),slice(200,300))], cand
print("base", [round((p24>t)[s].float().mean().item(),2) for s in (slice(0,100),slice(100,200),slice(200,300))])
for name,key in [("sO|sO>sH",lambda a,b:torch.where(a>b,a,torch.zeros_like(a))),("sO-sH",lambda a,b:a-b),("sO-0.5sH",lambda a,b:a-0.5*b),("sO",lambda a,b:a)]:
    for N in [100,300,1000,3000]:
        r,_=evalsel(key,N); print(name,N,"pred refusal O/C/H",r)
