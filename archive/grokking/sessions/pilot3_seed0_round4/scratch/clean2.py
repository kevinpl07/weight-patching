import numpy as np, sys
sys.path.insert(0, "<project>/runs/seed0/loop2/round4/agent/scratch")
from clean import *
def logits(W):
    w1, b1, w2, b2 = W["fc1.weight"], W["fc1.bias"], W["fc2.weight"], W["fc2.bias"]
    H = np.maximum(0, w1[:, A].T + w1[:, p + B].T + b1); return H @ w2.T + b2
def acc(L): return round(float((L.argmax(1) == Y).mean()), 4)
W = load(); C = clean(W)
Lf, Lc = logits(W), logits(C)
print("offset std full %.3f clean %.3f" % (Lf.mean(0).std(), Lc.mean(0).std()))
print("clean with full offset", acc(Lc - Lc.mean(0) + Lf.mean(0)))
R = Lf - Lc
print("residue logits std %.3f, residue offset std %.3f" % (R.std(), R.mean(0).std()))
# residue on correct class vs mean
print("residue correct-class boost", (R[np.arange(len(Y)), Y] - R.mean(1)).mean())
