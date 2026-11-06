import numpy as np
N = 256; g = 2 * np.pi * np.arange(N) / N
for beta in [-1.5, -1.0, -0.6, -0.3, 0, 0.3, 0.6, 1.0]:
    f = np.maximum(0, np.cos(g)[:, None] + np.cos(g)[None, :] + beta)
    F = np.fft.fft2(f) / N ** 2
    s = abs(F[1, 1]); lin = abs(F[1, 0]); am = abs(F[1, -1]); dc = abs(F[0, 0]); f20 = abs(F[2, 0])
    junk = np.sqrt(2 * lin ** 2 + 2 * am ** 2 + 2 * f20 ** 2)
    print("beta %+.1f  F11 %.3f  F10 %.3f  F1-1 %.3f  F00 %.3f F20 %.3f  junk(no dc)/sig %.2f  dc/sig %.2f  act %.2f" % (beta, s, lin, am, dc, f20, junk / s, dc / s, (f > 0).mean()))
