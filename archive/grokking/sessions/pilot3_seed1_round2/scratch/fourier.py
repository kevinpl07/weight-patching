import numpy as np, sys
S = sys.argv[1] if len(sys.argv)>1 else "."
W = dict(np.load(S + "/" + (sys.argv[2] if len(sys.argv)>2 else "weights.npz")))
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
A=W1[:,:p]; B=W1[:,p:]; O=W2.T  # O: [128,97]
def spec(X):
    F=np.fft.rfft(X,axis=1)  # k=0..48
    return F
FA,FB,FO=spec(A),spec(B),spec(O)
PA=np.abs(FA)**2; PB=np.abs(FB)**2; PO=np.abs(FO)**2
tot=PA[:,1:]+PB[:,1:]+PO[:,1:]
k=np.argmax(tot,axis=1)+1
rows=[]
for n in range(128):
    kk=k[n]
    fa=PA[n,kk]/PA[n,1:].sum(); fb=PB[n,kk]/PB[n,1:].sum(); fo=PO[n,kk]/PO[n,1:].sum()
    pha=np.angle(FA[n,kk]); phb=np.angle(FB[n,kk]); pho=np.angle(FO[n,kk])
    # for cos(w(a)+...) A ~ cos(2pi k a/p + phiA): rfft phase = phiA; output ~cos(2pi k c/p + phiO) ; want phiO = phiA+phiB
    err=np.angle(np.exp(1j*(pho-pha-phb)))
    rows.append((n,kk,fa,fb,fo,err,b1[n]))
print("n k fracA fracB fracO phase_err bias")
for r in rows: print("%3d %2d %.2f %.2f %.2f %+.2f %+.3f"%r)
from collections import Counter
print(Counter(k.tolist()))
# global freq power
print("global power by k (A,B,O):")
gp=(PA+PB+PO)[:,1:].sum(0)
order=np.argsort(-gp)
for kk in order[:12]: print(kk+1, round(gp[kk]/gp.sum(),3))
