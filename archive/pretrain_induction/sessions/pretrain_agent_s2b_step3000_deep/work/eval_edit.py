import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
h0 = int(sys.argv[1]) if len(sys.argv)>1 else 5
h1 = int(sys.argv[2]) if len(sys.argv)>2 else 5
x = T.train_windows(32, 512, seed=11)
lb = torch.cat([T.run(x[i:i+16], edited=False)["loss"] for i in range(0,32,16)]).mean().item()
le = torch.cat([T.run(x[i:i+16], edited=True)["loss"] for i in range(0,32,16)]).mean().item()
print(f"32x512 seed11: base {lb:.4f} edited {le:.4f} delta {le-lb:+.4f}")
x2 = T.train_windows(16, 512, seed=21)
cat = torch.load("work/diag3_cats.pt")["cat"]
ob = T.run(x2, edited=False, attn=True); oe = T.run(x2, edited=True, attn=True)
for c in range(3):
    m = cat==c
    print(f" cat{c}: base {ob['loss'][m].mean():.3f} edited {oe['loss'][m].mean():.3f} d {oe['loss'][m].mean()-ob['loss'][m].mean():+.3f}")
A0 = oe["attn"][0][:, h0]; t = torch.arange(1,512)
print(f"new L0H{h0}: self {A0[:, t, t].mean():.3f} prev {A0[:, t, t-1].mean():.3f}")
A1e = oe["attn"][1][:, h1]; A1b = ob["attn"][1][:, 2]
print(f"new L1H{h1} vs base L1H2 attention: L1 dist {(A1e-A1b).abs().sum(-1).mean():.3f}")
print(T.metrics(edited=True))
