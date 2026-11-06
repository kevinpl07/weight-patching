import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W=T.base_weights(); E=W["tok.weight"]; P=W["pos.weight"]
d=torch.load("work/r1_seed2.pt"); x=d["x"]; ho0=d["ho0"]  # [B,H,T,d]
H0=6
cnt=torch.bincount(T.train_windows(256,512,seed=50).reshape(-1),minlength=8192).float()
wfreq=cnt/cnt.sum()
Ev=E.var(0); Pv=P.var(0)
C=torch.argsort(Ev+Pv)[:32]
print("sin coords E var", Ev[C].mean().item(), "vs avg", Ev.mean().item())
mask=torch.ones(512,dtype=torch.bool); mask[C]=False
S0=torch.eye(512)[mask]            # 480x512 basis of non-C coords
L0=(ho0.sum(1)-ho0[:,H0]).reshape(-1,512); L0=L0-L0.mean(0)
CL=S0@(L0.T@L0/L0.shape[0])@S0.T
ev,evec=torch.linalg.eigh(CL)       # ascending
print("L0-out eig (480) quantiles", [round(ev[i].item(),5) for i in (0,100,200,300,400,479)])
for nq in (192,256,320):
  Qz=(evec[:,:nq].T@S0)               # nq x 512 orthonormal
  # token covariance (freq weighted, LN-normalized embeddings)
  En=F.layer_norm(E,(512,))
  mu=(wfreq[:,None]*En).sum(0)
  Ec=En-mu
  CE=(Ec.T*wfreq)@Ec
  CEq=Qz@CE@Qz.T
  e2,v2=torch.linalg.eigh(CEq)
  R=(v2[:,-64:].T@Qz)                 # 64x512 read basis
  Wr=(v2[:,:64].T@Qz)                 # lowest E var in quiet
  print(nq,"E var captured by R",round((R@CE@R.T).trace().item()/CE.trace().item(),3),"by Wr",round((Wr@CE@Wr.T).trace().item()/CE.trace().item(),4),
        "L0 var in R",round((R@(L0.T@L0/L0.shape[0])@R.T).trace().item(),4),"L0 total",round((L0.var(0).sum()).item(),2))
  # retrieval test on resid1 (layer-1 LN)
  z1=F.layer_norm(d["r1"],(512,)).reshape(-1,512); xt=x.reshape(-1)
  cand=torch.unique(x)
  for whiten in (False,True):
    Bq=R if not whiten else torch.diag(e2[-64:]**-0.5)@R
    a=z1@Bq.T; b=En[cand]@Bq.T
    acc=(cand[(a@b.T).argmax(-1)]==xt).float().mean().item()
    b0=F.layer_norm(E[x].reshape(-1,512),(512,))@Bq.T
    print("   whiten",whiten,"retrieval acc from resid1",round(acc,3), "from clean emb",round((cand[(b0@b.T).argmax(-1)]==xt).float().mean().item(),3))
