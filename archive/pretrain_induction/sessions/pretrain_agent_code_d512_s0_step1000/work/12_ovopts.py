import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
sys.path.insert(0,"work"); import build as B
W=T.base_weights(); s=B.stats(); E=W["tok.weight"]; U=W["head.weight"]
Up=U*W["ln_f.weight"][None]; En=s["En"]; wf=s["wf"]
top=torch.argsort(wf,descending=True)[:3000]
def q(Mfull, Vread, name, tw=None):
    codes=En[top]@Vread.T; lg=(codes@Mfull.T)@Up.T; lg=lg-lg.mean(-1,keepdim=True)
    own=lg[torch.arange(len(top)),top]; rank=(lg>own[:,None]).sum(-1).float()
    z=own/lg.std(-1)
    tw=wf[top]/wf[top].sum()
    print(f"{name:35s} rank med {rank.median().item():6.0f} top1 {(rank==0).float().mean().item():.3f} top10 {(rank<10).float().mean().item():.3f} z {z.mean().item():.2f} freq-weighted z {(z*tw).sum().item():.2f}")
for wm in ("uni","sqrt"):
    ww={"sqrt":wf.sqrt(),"uni":(wf>0).float()}[wm]; ww=ww/ww.sum()
    Upc=Up-(ww[:,None]*Up).sum(0); Enc=En-(ww[:,None]*En).sum(0)
    A=(Upc.T*ww)@Enc            # 512x512 hebbian
    q(A,torch.eye(512),f"{wm} full hebbian")
    Uu,S,Vh=torch.linalg.svd(A); 
    q(Uu[:,:64]*S[:64],Vh[:64],f"{wm} hebbian rank64 svd")
    R=s["R"]; q(A@R.T,R,f"{wm} hebbian via R (quiet)")
    # SVD restricted to R? same as above. Try top-64 SVD of A restricted to quiet subspace(192)
