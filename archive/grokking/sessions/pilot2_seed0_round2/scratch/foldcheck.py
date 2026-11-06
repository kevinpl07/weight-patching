import numpy as np
exec(open("<project>/runs/seed0/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
C=clean(W); print("clean",fwd(C))
F={k:v.copy() for k,v in C.items()}; F["fc2.bias"]+=((w2-C["fc2.weight"])*mact[None,:])[:,four].sum(1); print("clean+fold",fwd(F))
