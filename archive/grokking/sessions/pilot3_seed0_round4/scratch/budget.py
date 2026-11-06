import numpy as np, sys
sys.path.insert(0, "<project>/runs/seed0/loop2/round4/agent/scratch")
from modes import info, freq_modes, coef, b2
ks = sorted(set(d["k"] for d in info.values()))
tot = {}; sig = 0
for k in ks:
    ns = [i for i in info if info[i]["k"] == k]
    G = freq_modes(ns)
    G[(0, 0)] = G[(0, 0)] + np.conj(coef(b2, k)) / 2
    sig += abs(G[(1, 1)])
    for m, v in G.items():
        if m == (1, 1): continue
        key = m if m in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1), (2, 0), (0, 2), (-1, -1)] else "other"
        tot[key] = tot.get(key, 0) + 2 * abs(v) ** 2
# remaining b2 at freqs not used
x = np.arange(97)
pw = 0
for k in range(1, 49):
    if k not in ks: pw += 2 * abs(coef(b2, k) / 2) ** 2
print("b2 at unused freqs var", pw, " b2 mean", b2.mean())
print("signal margin 2*sum|G11| =", 2 * sig)
for k, v in sorted(tot.items(), key=lambda t: -t[1]): print(k, round(v, 3))
print("total junk var", sum(tot.values()), "std", np.sqrt(sum(tot.values())))
