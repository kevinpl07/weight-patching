from common import *
x = T.train_windows(32, 512, seed=5); C = cats(x)
b = T.run(x, edited=False)["loss"]; report(b, C, "base")
for c in [0.95, 1.05]:
    lo = eval_edit({"ln_f.weight": W["ln_f.weight"]*c, "ln_f.bias": W["ln_f.bias"]*c}, x)["loss"]; report(lo, C, f"ln_f x{c}")
