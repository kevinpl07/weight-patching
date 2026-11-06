from common import *
x = T.train_windows(32, 512, seed=5); C = cats(x)
b = T.run(x, edited=False)["loss"]; report(b, C, "base")
for cq in [0.9, 1.15, 1.3, 1.5]:
    q = W["blocks.1.attn.qkv.weight"].clone(); q[qkv_rows(0,2)] *= cq
    lo = eval_edit({"blocks.1.attn.qkv.weight": q}, x)["loss"]; report(lo, C, f"L1H2 Q x{cq}")
for cv in [0.9, 1.15, 1.3]:
    q = W["blocks.1.attn.qkv.weight"].clone(); q[qkv_rows(2,2)] *= cv
    lo = eval_edit({"blocks.1.attn.qkv.weight": q}, x)["loss"]; report(lo, C, f"L1H2 V x{cv}")
