from common import *
x = T.train_windows(32, 512, seed=5); C = cats(x)
print("cat fractions", [round((C==i).float().mean().item(),3) for i in range(5)])
b = T.run(x, edited=False)["loss"]; report(b, C, "base")
for l in range(2):
    for h in range(8):
        ed = {f"blocks.{l}.attn.proj.weight": W[f"blocks.{l}.attn.proj.weight"].clone()}
        ed[f"blocks.{l}.attn.proj.weight"][:, head_cols(h)] = 0
        lo = eval_edit(ed, x, f"ablate L{l}H{h}")["loss"]; report(lo, C, f"abl L{l}H{h} d={lo.mean()-b.mean():+.3f}")
