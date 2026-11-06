"""Variant A: L0H{h} := self-attention head with a complementary token code in V and ZERO output projection.
Gives training a second fast-learning token->residual map (rank 64) next to L0H4."""
import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F, json
W = T.base_weights(); d=512; V=8192
h = int(sys.argv[1]) if len(sys.argv)>1 else 5
oinit = sys.argv[2] if len(sys.argv)>2 else "zero"
xs = T.train_windows(2000, 512, seed=7000)
freq = torch.bincount(xs.reshape(-1), minlength=V).float() + 1; freq = freq/freq.sum()
qkv0 = W["blocks.0.attn.qkv.weight"].clone(); P0 = W["blocks.0.attn.proj.weight"].clone()
Z = F.layer_norm(W["tok.weight"] + W["pos.weight"].mean(0), (d,), W["blocks.0.ln1.weight"], W["blocks.0.ln1.bias"])
wt = freq**0.5; wt = wt/wt.sum()
Zc = Z - (wt[:,None]*Z).sum(0)
Wv4 = qkv0[1024+4*64:1024+5*64]
Q4,_ = torch.linalg.qr(Wv4.T)
Zp = Zc - (Zc@Q4)@Q4.T
ev, evec = torch.linalg.eigh((Zp*wt[:,None]).T @ Zp)
U = evec[:, -64:].flip(1); sd = ev[-64:].flip(0).sqrt()
# scale V so that |v| matches L0H4's typical |v| (frequency weighted)
v4 = (Zc @ Wv4.T); n4 = ((v4.pow(2).sum(1))*freq).sum().sqrt()
Vr = (U/sd).T
vn = ((Zc @ Vr.T).pow(2).sum(1)*freq).sum().sqrt()
Vr = Vr * (n4/vn)
print("L0H4 |v| %.2f ; new |v| scaled to match" % n4)
K6 = qkv0[512+6*64:512+7*64]
sl = slice(h*64, h*64+64)
qkv0[0:512][sl] = K6; qkv0[512:1024][sl] = K6; qkv0[1024:1536][sl] = Vr
if oinit == "zero":
    P0[:, sl] = 0
T.save_edit({"blocks.0.attn.qkv.weight": qkv0, "blocks.0.attn.proj.weight": P0}, note=f"variant A h={h} O={oinit}")
