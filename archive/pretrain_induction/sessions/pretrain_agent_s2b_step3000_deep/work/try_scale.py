import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
W = T.base_weights()
x = T.train_windows(32, 512, seed=11)
def ev():
    return torch.cat([T.run(x[i:i+16], edited=True)["loss"] for i in range(0,32,16)]).mean().item()
P0 = W["blocks.0.attn.proj.weight"]; P1 = W["blocks.1.attn.proj.weight"]
def colscale(P, heads, c):
    P = P.clone()
    for h in heads: P[:, h*64:(h+1)*64] *= c
    return P
for name, e in [
    ("L0proj*0.5", {"blocks.0.attn.proj.weight": P0*0.5}),
    ("L1proj*0.5", {"blocks.1.attn.proj.weight": P1*0.5}),
    ("L0H3*0.5", {"blocks.0.attn.proj.weight": colscale(P0,[3],0.5)}),
    ("L0H3*0.25", {"blocks.0.attn.proj.weight": colscale(P0,[3],0.25)}),
    ("L0proj*0.5+L1 H0-3*0.5", {"blocks.0.attn.proj.weight": P0*0.5, "blocks.1.attn.proj.weight": colscale(P1,[0,1,2,3],0.5)}),
    ]:
    T.save_edit(e, note=name)
    print(name, round(ev(),4))
T.save_edit({}, note="empty")
