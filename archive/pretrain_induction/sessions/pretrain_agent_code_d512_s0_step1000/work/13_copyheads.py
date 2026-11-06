import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
sys.path.insert(0,"work"); import build as B
W=T.base_weights(); s=B.stats(); U=W["head.weight"]
Up=U*W["ln_f.weight"][None]; En=s["En"]; wf=s["wf"]
top=torch.argsort(wf,descending=True)[:3000]; tw=wf[top]/wf[top].sum()
for l in (0,1):
  qkv=W[f"blocks.{l}.attn.qkv.weight"]; pr=W[f"blocks.{l}.attn.proj.weight"]; lw=W[f"blocks.{l}.ln1.weight"]
  for h in range(8):
    Wv=qkv[1024+h*64:1024+h*64+64]; Wo=pr[:,h*64:h*64+64]
    lg=((En[top]*lw)@Wv.T@Wo.T)@Up.T; lg=lg-lg.mean(-1,keepdim=True)
    own=lg[torch.arange(len(top)),top]; rank=(lg>own[:,None]).sum(-1).float(); z=own/lg.std(-1)
    print(f"L{l}H{h} copy: rank med {rank.median().item():6.0f} top10 {(rank<10).float().mean().item():.3f} z {z.mean().item():+.2f} fw-z {(z*tw).sum().item():+.2f}")
