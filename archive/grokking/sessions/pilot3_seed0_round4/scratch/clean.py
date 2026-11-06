import numpy as np, sys
sys.path.insert(0, "<project>/runs/seed0/loop2/round4/agent/scratch")
S = "<project>/runs/seed0/loop2/round4/agent"
p = 97; x = np.arange(p)
d = np.load(S + "/train_pairs.npy"); A, B, Y = d.T
def load(f="working.npz"): return {k: v.copy() for k, v in np.load(S + "/" + f).items()}
def mainfreq(v3):
    tot = sum(np.abs(np.fft.rfft(v))[1:] ** 2 for v in v3); return int(np.argmax(tot) + 1)
def proj(v, k):
    c = 2 / p * (v * np.exp(-2j * np.pi * k * x / p)).sum()
    return v.mean() + np.real(c * np.exp(2j * np.pi * k * x / p))
def live_idx(W):
    w1, b1 = W["fc1.weight"], W["fc1.bias"]
    H = np.maximum(0, w1[:, A].T + w1[:, p + B].T + b1); return np.where((H > 0).mean(0) > 0)[0]
def clean(W):
    C = {k: v.copy() for k, v in W.items()}
    for i in live_idx(W):
        k = mainfreq((W["fc1.weight"][i, :p], W["fc1.weight"][i, p:], W["fc2.weight"][:, i]))
        C["fc1.weight"][i, :p] = proj(W["fc1.weight"][i, :p], k)
        C["fc1.weight"][i, p:] = proj(W["fc1.weight"][i, p:], k)
        C["fc2.weight"][:, i] = proj(W["fc2.weight"][:, i], k)
    return C
def evaluate(W, center=False):
    w1, b1, w2, b2 = W["fc1.weight"], W["fc1.bias"], W["fc2.weight"], W["fc2.bias"]
    H = np.maximum(0, w1[:, A].T + w1[:, p + B].T + b1)
    L = H @ w2.T + b2
    if center: L = L - L.mean(0)
    L2 = L - L.max(1, keepdims=True); lp = L2 - np.log(np.exp(L2).sum(1, keepdims=True))
    return round(float((L.argmax(1) == Y).mean()), 4), round(float(-lp[np.arange(len(Y)), Y].mean()), 4), round(float((L[np.arange(len(Y)), Y] - L.mean(1)).mean()), 2)
if __name__ == "__main__":
    W = load()
    print("full", evaluate(W)); C = clean(W)
    print("clean", evaluate(C), "clean centered", evaluate(C, True))
