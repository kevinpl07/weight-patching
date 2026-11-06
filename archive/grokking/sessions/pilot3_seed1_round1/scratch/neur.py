import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; w2=W["fc2.weight"]; b1=W["fc1.bias"]
A=w1[:,:p]; B=w1[:,p:]; O=w2.T
def info(M):
    F=np.fft.rfft(M,axis=1); P=np.abs(F[:,1:])**2; Pn=P/P.sum(1,keepdims=True)
    k=Pn.argmax(1); return k+1, Pn[np.arange(len(k)),k], np.angle(F[np.arange(len(k)),k+1])
ka,sa,pa=info(A); kb,sb,pb=info(B); ko,so,po=info(O)
order=np.argsort(-sa)
for i in order:
    print(f"n{i:3d} a k{ka[i]:2d} {sa[i]:.2f} | b k{kb[i]:2d} {sb[i]:.2f} | out k{ko[i]:2d} {so[i]:.2f} | bias {b1[i]:+.3f} | phase a {pa[i]:+.2f} b {pb[i]:+.2f} o {po[i]:+.2f}")
