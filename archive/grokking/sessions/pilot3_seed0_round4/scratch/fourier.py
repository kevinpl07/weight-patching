import numpy as np, sys
S = "<project>/runs/seed0/loop2/round4/agent"
W = dict(np.load(S + "/" + (sys.argv[1] if len(sys.argv) > 1 else "working.npz")))
p = 97
w1, b1, w2, b2 = W["fc1.weight"], W["fc1.bias"], W["fc2.weight"], W["fc2.bias"]
data = np.load(S + "/train_pairs.npy")
A, B, Y = data[:, 0], data[:, 1], data[:, 2]
H = np.maximum(0, w1[:, A].T + w1[:, p + B].T + b1)
act = (H > 0).mean(0)
live = np.where(act > 0)[0]
print("live", len(live), "dead", 128 - len(live))
print("dead max |w1|", np.abs(w1[act == 0]).max(), "bias range", b1[act == 0].min(), b1[act == 0].max(), "max|w2| dead", np.abs(w2[:, act == 0]).max())
F = np.fft.rfft
def spec(v):
    f = np.fft.rfft(v); pw = np.abs(f) ** 2; pw[1:] *= 2
    return f, pw / pw.sum()
rows = []
for i in live:
    fa, pa = spec(w1[i, :p]); fb, pb = spec(w1[i, p:]); fo, po = spec(w2[:, i])
    tot = pa + pb + po
    k = np.argmax(tot[1:]) + 1
    phA = np.angle(fa[k]); phB = np.angle(fb[k]); phO = np.angle(fo[k])
    rows.append((k, i, pa[k], pb[k], po[k], phA, phB, phO, act[i], b1[i], np.abs(fa[k]) * 2 / p, np.abs(fb[k]) * 2 / p, np.abs(fo[k]) * 2 / p))
rows.sort()
print(" k  neur  fracA fracB fracO   phA    phB    phO   act   bias  ampA ampB ampO")
for r in rows:
    print("%2d %4d  %.2f %.2f %.2f  %+.2f %+.2f %+.2f  %.2f %+.3f %.2f %.2f %.2f" % r)
from collections import Counter
print(Counter(r[0] for r in rows))
# fc2 bias spectrum
fb2, pb2 = spec(b2); top = np.argsort(pb2)[::-1][:8]; print("b2 spectrum top", [(int(t), round(float(pb2[t]), 3)) for t in top])
logits = H @ w2.T + b2
print("train acc", (logits.argmax(1) == Y).mean())
