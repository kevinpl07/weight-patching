import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
sys.path.insert(0,"work"); import build as B
W=T.base_weights(); s=B.stats(); U=W["head.weight"]
Up=U*W["ln_f.weight"][None]; En=s["En"]; R=s["R"]; wf=s["wf"]
d=torch.load("work/r1_seed2.pt"); x=d["x"][:2]; r1=d["r1"][:2]-d["ho0"][:2,B.H0]
z1=F.layer_norm(r1,(512,),W["blocks.1.ln1.weight"],W["blocks.1.ln1.bias"]).reshape(-1,512); xt=x.reshape(-1)
lw1=W["blocks.1.ln1.weight"]
ww=(wf>0).float(); ww=ww/ww.sum()
Upc=Up-(ww[:,None]*Up).sum(0)
codes=(En-(ww[:,None]*En).sum(0))@R.T
CU=(Upc.T*ww)@Upc; eu,vu=torch.linalg.eigh(CU)
CC=(codes.T*ww)@codes; ec,vc=torch.linalg.eigh(CC)
v=z1@(R/lw1[None]).T
for ku in (0,8,32,64):
  for kc in (0,4,16):
    Pu=torch.eye(512)-vu[:,512-ku:]@vu[:,512-ku:].T if ku else torch.eye(512)
    Pc=torch.eye(64)-vc[:,64-kc:]@vc[:,64-kc:].T if kc else torch.eye(64)
    M=Pu@(Upc.T*ww)@codes@Pc
    lg=(v@M.T)@Up.T; lg=lg-lg.mean(-1,keepdim=True)
    own=lg[torch.arange(len(xt)),xt]; rank=(lg>own[:,None]).sum(-1).float(); z=own/lg.std(-1)
    print(f"remove U-PCs {ku:3d} code-PCs {kc:3d}: rank med {rank.median().item():5.0f} top1 {(rank==0).float().mean().item():.3f} top10 {(rank<10).float().mean().item():.3f} z {z.mean().item():.2f}")
