import numpy as np, sys
S = "<project>/runs/seed0/loop2/round4/agent"
W = dict(np.load(S + "/" + "weights.npz"))
p = 97
w1, b1, w2, b2 = W["fc1.weight"], W["fc1.bias"], W["fc2.weight"], W["fc2.bias"]
data = np.load(S + "/train_pairs.npy"); A, B = data[:, 0], data[:, 1]
H = np.maximum(0, w1[:, A].T + w1[:, p + B].T + b1); act = (H > 0).mean(0)
live = np.where(act > 0)[0]
N = 128; g = 2 * np.pi * np.arange(N) / N
def coef(v, k):  # complex amp: v ~ Re(c e^{i theta k x}) -> c = 2/p sum v e^{-i..}
    x = np.arange(p); return 2 / p * (v * np.exp(-2j * np.pi * k * x / p)).sum()
def mainfreq(i):
    tot = sum(np.abs(np.fft.rfft(v))[1:] ** 2 for v in (w1[i, :p], w1[i, p:], w2[:, i]))
    return int(np.argmax(tot) + 1)
info = {}
for i in live:
    k = mainfreq(i)
    ca, cb, co = coef(w1[i, :p], k), coef(w1[i, p:], k), coef(w2[:, i], k)
    da, db = w1[i, :p].mean(), w1[i, p:].mean()
    info[i] = dict(k=k, Aa=abs(ca), pa=np.angle(ca), Ab=abs(cb), pb=np.angle(cb), O=abs(co), po=np.angle(co), beta=b1[i] + da + db)
def neuron_modes(Aa, Ab, beta, M=3):
    f = np.maximum(0, Aa * np.cos(g)[:, None] + Ab * np.cos(g)[None, :] + beta)
    F = np.fft.fft2(f) / N ** 2  # f = sum F[m,n] e^{i(mu+nv)}
    return F
def freq_modes(neurons):
    G = {}
    for i in neurons:
        d = info[i]; F = neuron_modes(d["Aa"], d["Ab"], d["beta"])
        for m in range(-3, 4):
            for n in range(-3, 4):
                # logit term: O cos(theta c + po) * F[m,n] e^{i(m(u)+n(v))}; coefficient of e^{-i theta c}: O/2 e^{-i po}
                G[(m, n)] = G.get((m, n), 0) + d["O"] / 2 * np.exp(-1j * d["po"]) * F[m % N, n % N] * np.exp(1j * (m * d["pa"] + n * d["pb"]))
    return G
if __name__ == "__main__":
    ks = sorted(set(d["k"] for d in info.values()))
    print(" k  n | signal(1,1) phase | (1,0)  (0,1)  (1,-1) (-1,1) (0,0)  (2,0) (0,2) | b2coef")
    for k in ks:
        ns = [i for i in info if info[i]["k"] == k]
        G = freq_modes(ns)
        s = G[(1, 1)]
        bc = coef(b2, k)
        print("%2d %2d | %.3f %+.2f | %.3f %.3f %.3f %.3f %.3f %.3f %.3f | %.3f  D+b2 %.3f" % (k, len(ns), abs(s), np.angle(s), abs(G[(1, 0)]), abs(G[(0, 1)]), abs(G[(1, -1)]), abs(G[(-1, 1)]), abs(G[(0, 0)]), abs(G[(2, 0)]), abs(G[(0, 2)]), abs(bc), abs(G[(0, 0)] + np.conj(bc) / 2)))
