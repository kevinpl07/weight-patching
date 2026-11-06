import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, math
x = T.train_windows(32, 512, seed=11)
outs = []
for i in range(0, 32, 16):
    outs.append(T.run(x[i:i+16], edited=False, attn=True, resid=True))
loss = torch.cat([o["loss"] for o in outs]); print("base loss %.4f"%loss.mean())
A = [torch.cat([o["attn"][l] for o in outs]) for l in range(2)]
HO = [torch.cat([o["head_out"][l] for o in outs]) for l in range(2)]
B,H,Tn,_ = A[0].shape
idx = torch.arange(Tn)
dist = (idx[:,None]-idx[None,:]).clamp(min=0).float()
for l in range(2):
    for h in range(8):
        p = A[l][:,h,64:]  # queries >=64
        ent = -(p.clamp_min(1e-12).log()*p).sum(-1).mean()
        md = (p*dist[64:]).sum(-1).mean()
        prof = [p.diagonal(offset=-o, dim1=-2, dim2=-1).mean().item() for o in range(0,6)]
        bos = p[...,0].mean()
        hn = HO[l][:,h].norm(dim=-1).mean()
        mx = p.max(-1).values.mean()
        print(f"L{l}H{h} ent {ent:.2f} maxp {mx:.2f} meandist {md:6.1f} pos0 {bos:.3f} offs0-5 "+" ".join(f"{v:.2f}" for v in prof)+f" |out| {hn:.2f}")
# head output redundancy: cosine between per-token head outputs (flattened)
for l in range(2):
    F_ = HO[l][:, :, 64:].reshape(B, 8, -1).permute(1,0,2).reshape(8,-1)
    F_ = F_ - 0
    C = torch.nn.functional.normalize(F_, dim=-1) @ torch.nn.functional.normalize(F_, dim=-1).T
    print(f"L{l} head-output cosine:\n", (C*100).round().int())
# residual norms
for i,r in enumerate(outs[0]["resid"]): print("resid",i,"norm %.2f"%r.norm(dim=-1).mean())
# mean ablation of each head (replace head output with its mean over batch&pos)
W = T.base_weights()
import copy
lnf_w, lnf_b, Wh = W["ln_f.weight"], W["ln_f.bias"], W["head.weight"]
def loss_from(res, toks):
    z = torch.nn.functional.layer_norm(res, (512,), lnf_w, lnf_b)
    lg = z[:, :-1] @ Wh.T
    return torch.nn.functional.cross_entropy(lg.transpose(1,2), toks[:,1:], reduction="none")
Rf = torch.cat([o["resid"][2] for o in outs])
l0 = loss_from(Rf[:8], x[:8]); print("recomputed loss %.4f (orig %.4f)"%(l0.mean(), loss[:8].mean()))
for h in range(8):
    ho = HO[1][:8,h]
    lz = loss_from(Rf[:8]-ho, x[:8]).mean()
    lm = loss_from(Rf[:8]-ho+ho.mean((0,1)), x[:8]).mean()
    print(f"ablate L1H{h}: zero +{lz-l0.mean():.4f} mean +{lm-l0.mean():.4f}")
# direct-path contribution of layer-0 heads (only direct effect)
for h in range(8):
    ho = HO[0][:8,h]
    lm = loss_from(Rf[:8]-ho+ho.mean((0,1)), x[:8]).mean()
    print(f"direct-path mean-ablate L0H{h}: +{lm-l0.mean():.4f}")
tok = T.tokenizer()
strs = [tok.decode([i]) for i in range(8192)]
tgt = x[:,1:]; prev = x[:,:-1]
def cat(t):
    s = strs[t]
    if t==0: return "eos"
    if s.strip()=="" and "\n" in s: return "nl+ws"
    if s.strip()=="": return "ws"
    if s.strip().isidentifier(): return "ident/kw"
    if any(c.isdigit() for c in s): return "num"
    if any(c in "\"'" for c in s): return "quote"
    return "punct"
cats = torch.tensor([["eos","nl+ws","ws","ident/kw","num","quote","punct"].index(cat(t)) for t in range(8192)])
names=["eos","nl+ws","ws","ident/kw","num","quote","punct"]
tc = cats[tgt]
tot = loss.sum()
for i,n in enumerate(names):
    m = tc==i
    print(f"target {n:9s} frac {m.float().mean():.3f} meanloss {loss[m].mean():.3f} share {loss[m].sum()/tot:.3f}")
pc = cats[prev]
for i,n in enumerate(names):
    m = pc==i
    print(f"after {n:9s} frac {m.float().mean():.3f} meanloss {loss[m].mean():.3f} share {loss[m].sum()/tot:.3f}")
