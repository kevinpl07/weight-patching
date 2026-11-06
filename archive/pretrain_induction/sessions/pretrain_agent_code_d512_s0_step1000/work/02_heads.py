import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W=T.base_weights()
x=T.train_windows(8,512,seed=1)
o=T.run(x,edited=False,attn=True,resid=True)
print("base loss",o["loss"].mean().item())
Tn=512
for l,a in enumerate(o["attn"]):
    # a [B,H,T,T]
    ent=-(a.clamp_min(1e-9).log()*a).sum(-1).mean((0,2))
    # mean attention distance
    idx=torch.arange(Tn)
    dist=(a*(idx[:,None]-idx[None,:]).clamp(min=0).float()).sum(-1).mean((0,2))
    diag0=a.diagonal(0,-2,-1).mean((0,2)); d1=a.diagonal(-1,-2,-1).mean((0,2)); d2=a.diagonal(-2,-2,-1).mean((0,2))
    first=a[...,0].mean((0,2))
    # attention to eos tokens
    for h in range(8):
        print(f"L{l}H{h} ent {ent[h]:.2f} dist {dist[h]:.1f} self {diag0[h]:.3f} prev {d1[h]:.3f} prev2 {d2[h]:.3f} first {first[h]:.3f} outnorm {o['head_out'][l][:,h].norm(dim=-1).mean():.3f}")
# ablate layer1 heads via head_out
lnf_w,lnf_b,U=W["ln_f.weight"],W["ln_f.bias"],W["head.weight"]
final=o["resid"][-1]
def loss_from(r):
    lg=F.layer_norm(r,(512,),lnf_w,lnf_b)@U.T
    return F.cross_entropy(lg[:,:-1].reshape(-1,8192),x[:,1:].reshape(-1)).item()
print("recomputed",loss_from(final))
for h in range(8):
    print("ablate L1H",h, round(loss_from(final-o["head_out"][1][:,h])-loss_from(final),4))
print("resid norms", [r.norm(dim=-1).mean().item() for r in o["resid"]])
