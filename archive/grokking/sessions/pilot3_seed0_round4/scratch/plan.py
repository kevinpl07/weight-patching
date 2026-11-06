import numpy as np, sys
sys.path.insert(0, "<project>/runs/seed0/loop2/round4/agent/scratch")
from cands import *
def quad_of(i, sign=+1, A=1.1, O=1.1):
    d = info[i]
    return dict(k=d["k"], Aa=A, pa=d["pa"] + sign * np.pi / 2, Ab=A, pb=d["pb"] + sign * np.pi / 2, O=O, po=d["pa"] + d["pb"] + np.pi, beta=0.0)
optA = [quad_of(i) for i in [30, 0, 3, 23, 76, 98, 36]]   # one quad per mirror pair + 28 + 18
optB = [quad_of(30, 1), quad_of(30, -1), quad_of(0, 1), quad_of(0, -1), quad_of(3, 1), quad_of(3, -1), quad_of(23, 1)]
optC = [quad_of(i) for i in [30, 0, 3, 23, 76]]
for n, o in [("base", []), ("A", optA), ("B", optB), ("C5", optC)]:
    print(n, score(o))
