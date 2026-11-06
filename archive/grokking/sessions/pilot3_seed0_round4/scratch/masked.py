import numpy as np
p = 97; t = 2 * np.pi * np.arange(p) / p
for thr in [0.0, 0.5, 0.7071, 0.85, 0.92]:
    g = np.cos(t) * (np.abs(np.cos(t)) > thr)
    frac = (np.abs(np.cos(t)) > thr).mean()
    f = np.maximum(0, g[:, None] + g[None, :])
    F = np.fft.fft2(f) / p ** 2
    G = np.abs(np.fft.fft(g)) / p
    print("thr %.2f frac %.2f  k-amp %.3f 3k-amp %.3f | F11 %.3f F10 %.3f F1-1 %.3f F00 %.3f F33 %.3f F31 %.3f" % (thr, frac, 2 * G[1], 2 * G[3], abs(F[1, 1]), abs(F[1, 0]), abs(F[1, -1]), abs(F[0, 0]), abs(F[3, 3]), abs(F[3, 1])))
