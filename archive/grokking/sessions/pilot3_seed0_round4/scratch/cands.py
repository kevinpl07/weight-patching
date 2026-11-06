import numpy as np, sys
sys.path.insert(0, "<project>/runs/seed0/loop2/round4/agent/scratch")
import modes
from modes import info, freq_modes, coef, b2, neuron_modes
p = 97
ks = sorted(set(d["k"] for d in info.values()))
def design(k, A=1.1, O=1.1, use_b2=False):
    ns = [i for i in info if info[i]["k"] == k]
    G = freq_modes(ns)
    g00 = G[(0, 0)] + (np.conj(coef(b2, k)) / 2 if use_b2 else 0)
    chi = -np.angle(-g00)
    delta = np.angle(-G[(1, -1)]) + chi
    best = None
    for flip in (0, np.pi):
        pa = (chi + delta) / 2 + flip; pb = (chi - delta) / 2 + flip
        d = dict(k=k, Aa=A, pa=pa, Ab=A, pb=pb, O=O, po=chi, beta=0.0)
        F = neuron_modes(A, A, 0.0)
        lin = abs(G[(1, 0)] + O / 2 * np.exp(-1j * chi) * F[1, 0] * np.exp(1j * pa)) ** 2 + abs(G[(0, 1)] + O / 2 * np.exp(-1j * chi) * F[0, 1] * np.exp(1j * pb)) ** 2
        if best is None or lin < best[0]: best = (lin, d)
    return best[1]
def score(extra):
    """extra: list of neuron dicts added. returns margin, junk std, ratio"""
    sig = 0; var = 0
    allk = sorted(set(ks) | set(d["k"] for d in extra))
    for k in allk:
        ns = [i for i in info if info[i]["k"] == k]
        G = freq_modes(ns) if ns else {}
        for j, d in enumerate(extra):
            if d["k"] != k: continue
            key = ("x", k, j); info[key] = d
            G2 = freq_modes([key]); del info[key]
            for m in G2: G[m] = G.get(m, 0) + G2[m]
        sig += abs(G[(1, 1)])
        var += sum(2 * abs(v) ** 2 for m, v in G.items() if m != (1, 1))
    return 2 * sig, np.sqrt(var), 2 * sig / np.sqrt(var)
if __name__ == "__main__":
    print("base", score([]))
    res = []
    for k in ks:
        d = design(k); res.append((score([d])[2], k))
    for r in sorted(res, reverse=True)[:12]: print("one quad at k=%d -> ratio %.4f" % (r[1], r[0]))
