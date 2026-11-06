import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
sys.path.insert(0,"work"); import build as B
W=T.base_weights(); s=B.stats(); U=W["head.weight"]
Up=U*W["ln_f.weight"][None]
d=torch.load("work/r1_seed2.pt"); x=d["x"][:2]; r1=d["r1"][:2]; ho0=d["ho0"][:2]
r1=r1-ho0[:,B.H0]          # remove old H6 output (we replace it)
z1=F.layer_norm(r1,(512,),W["blocks.1.ln1.weight"],W["blocks.1.ln1.bias"]).reshape(-1,512); xt=x.reshape(-1)
lw1=W["blocks.1.ln1.weight"]
C=s["C"]; R=s["R"]; Wr=s["Wr"]
# quiet subspace minus Wr: rebuild Qz
Qz=None
import math
def evalOV(Wv,Wo,name):
    v=z1@Wv.T; res=v@Wo.T
    lg=res@Up.T; lg=lg-lg.mean(-1,keepdim=True)
    own=lg[torch.arange(len(xt)),xt]; rank=(lg>own[:,None]).sum(-1).float(); z=own/lg.std(-1)
    print(f"{name:40s} on real resid1: rank med {rank.median().item():6.0f} top10 {(rank<10).float().mean().item():.3f} z {z.mean().item():+.2f}")
qkv0=W["blocks.0.attn.qkv.weight"]; pr0=W["blocks.0.attn.proj.weight"]; lw0=W["blocks.0.ln1.weight"]
# projector onto complement of (layer-0 output top dims, Wr, C coords) : use quiet basis
cache=B._cache
mask=torch.ones(512,dtype=torch.bool); mask[C]=False; S0=torch.eye(512)[mask]
L0=(d["ho0"].sum(1)-d["ho0"][:,B.H0]).reshape(-1,512); L0=L0-L0.mean(0)
ev,evec=torch.linalg.eigh(S0@(L0.T@L0/L0.shape[0])@S0.T)
for nq in (128,192,256,512):
    Qz=evec[:,:nq].T@S0 if nq<512 else torch.eye(512)
    Pi=Qz.T@Qz; Pi=Pi-Wr.T@Wr@Pi if nq<512 else Pi
    for h in (2,6):
        Wv=qkv0[1024+h*64:1024+h*64+64]*(lw0/lw1)[None]; Wo=pr0[:,h*64:h*64+64]
        evalOV(Wv@Pi,Wo,f"L0H{h} OV transplant, read quiet{nq}")
ww=(s["wf"]>0).float(); ww=ww/ww.sum(); Upc=Up-(ww[:,None]*Up).sum(0); Enc=s["En"]-(ww[:,None]*s["En"]).sum(0)
evalOV(R/lw1[None],(Upc.T*ww)@(Enc@R.T),"hebbian via R")
