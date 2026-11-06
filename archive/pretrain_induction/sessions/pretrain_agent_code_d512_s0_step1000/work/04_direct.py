import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W=T.base_weights()
E=W["tok.weight"]; P=W["pos.weight"]
x=T.train_windows(8,512,seed=2)
o=T.run(x,edited=False,resid=True)
r0,r1=o["resid"][0],o["resid"][1]
torch.save(dict(x=x,r1=r1,ho0=o["head_out"][0]),"work/r1_seed2.pt")
Ec=E-E.mean(0)
U,S,Vh=torch.linalg.svd(Ec,full_matrices=False)
for k in (64,):
    D=Vh[:k]
    e=E[x]                       # [B,T,d]
    for name,r in [("embed",r0),("resid1",r1)]:
        z=F.layer_norm(r,(512,))
        a=z@D.T; b=F.layer_norm(e,(512,))@D.T
        cos=F.cosine_similarity(a,b,dim=-1).mean().item()
        # retrieval among 2000 frequent tokens in batch
        cand=torch.unique(x)
        cb=F.layer_norm(E[cand],(512,))@D.T
        sc=a.reshape(-1,k)@cb.T
        pred=cand[sc.argmax(-1)]
        acc=(pred==x.reshape(-1)).float().mean().item()
        print(name,k,"cos",round(cos,3),"retrieval acc",round(acc,3),"ncand",len(cand))
# how much of resid1 is in the E-PCA dirs vs other
D=Vh[:64]
for name,r in [("embed",r0),("resid1",r1),("ho0sum",o["head_out"][0].sum(1))]:
    rc=r-r.mean((0,1))
    print(name,"var frac in top64 E dirs",round(((rc@D.T)**2).sum().item()/(rc**2).sum().item(),3),"total var/dim",round(rc.var().item(),4))
# resid1 PCA spectrum
rc=(r1-r1.mean((0,1))).reshape(-1,512)
s=torch.linalg.svdvals(rc); print("resid1 energy top 16/64/256",[round((s[:k]**2).sum().item()/(s**2).sum().item(),3) for k in (16,64,256)])
