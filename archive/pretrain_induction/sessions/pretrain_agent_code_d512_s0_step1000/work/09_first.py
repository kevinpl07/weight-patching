import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
sys.path.insert(0,"work"); import build as B
W=T.base_weights()
o=B.build(**eval(sys.argv[1]) if len(sys.argv)>1 else {})
T.save_edit(o,note="v0 calibration build")
x=T.train_windows(8,512,seed=5)
r=T.run(x,attn=True,resid=True)
a0=r["attn"][0][:,B.H0]
print("L0H6 prev",a0.diagonal(-1,-2,-1).mean().item(),"self",a0.diagonal(0,-2,-1).mean().item(),"prev2",a0.diagonal(-2,-2,-1).mean().item())
print("L0H6 out norm",r["head_out"][0][:,B.H0].norm(dim=-1).mean().item(), "resid1 norm", r["resid"][1].norm(dim=-1).mean().item())
z1=F.layer_norm(r["resid"][1],(512,),W["blocks.1.ln1.weight"],W["blocks.1.ln1.bias"])
qkv=o["blocks.1.attn.qkv.weight"]; sl=slice(B.H1*64,B.H1*64+64)
q=z1@qkv[sl].T; k=z1@qkv[512:][sl].T
S=(q@k.transpose(1,2))/8
Tn=512; causal=torch.ones(Tn,Tn).tril().bool()
match=(x[:,None,:-1]==x[:,:,None])  # [B,t,s-1] : x[s-1]==x[t] for s=1..
M=torch.zeros(8,Tn,Tn,dtype=torch.bool); M[:,:,1:]=match
M&=causal
NM=causal.clone()[None].expand(8,-1,-1)&~M
print("score match mean",S[M].mean().item(),"nonmatch mean",S[NM].mean().item(),"nonmatch std",S[NM].std().item(), "nonmatch q99", S[NM][::50].quantile(0.99).item())
print("frac queries with a match", M.any(-1)[:,1:].float().mean().item())
print("loss",r["loss"].mean().item())
