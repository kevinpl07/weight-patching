import numpy as np, sys
D="<project>/runs/seed1/loop2/round4/agent/"
W=dict(np.load(D+(sys.argv[1] if len(sys.argv)>1 else "weights.npz")))
p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
A=W1[:,:p];B=W1[:,p:]
F=np.exp(-2j*np.pi*np.outer(np.arange(p),np.arange(p))/p)
def spec(v):
    c=v@F.T  # (n,p)
    pw=np.abs(c[:,1:49])**2*2
    return c,pw
cA,pA=spec(A);cB,pB=spec(B);cO,pO=spec(W2.T)
tot=lambda v:((v-v.mean(1,keepdims=True))**2).sum(1)
alive=np.abs(W2).sum(0)>0.1
print("alive",alive.sum(), "b1 dead mean",b1[~alive].mean() if (~alive).any() else None)
print("dead weight norms max", np.abs(W1[~alive]).max(), np.abs(W2[:,~alive]).max())
rows=[]
for n in np.where(alive)[0]:
    k=np.argmax(pA[n]+pB[n]+pO[n])+1
    fa=pA[n,k-1]/tot(A[n:n+1])[0]; fb=pB[n,k-1]/tot(B[n:n+1])[0]; fo=pO[n,k-1]/tot(W2.T[n:n+1])[0]
    pa=np.angle(cA[n,k]);pb=np.angle(cB[n,k]);po=np.angle(cO[n,k])
    err=np.angle(np.exp(1j*(-po-(pa+pb)))) # check
    err2=np.angle(np.exp(1j*(po-(pa+pb))))
    rows.append((n,k,fa,fb,fo,abs(cA[n,k])*2/p,abs(cB[n,k])*2/p,abs(cO[n,k])*2/p,err,err2,b1[n],A[n].mean(),B[n].mean()))
rows.sort(key=lambda r:r[1])
for r in rows: print("n%3d k%2d fa%.2f fb%.2f fo%.2f |A|%.2f |B|%.2f |O|%.2f e%+.2f e2%+.2f b%+.2f mA%+.2f mB%+.2f"%r)
from collections import Counter
print(Counter(r[1] for r in rows))
print("W1 abs mean alive",np.abs(W1[alive]).mean(),"W2",np.abs(W2[:,alive]).mean(),"b2",b2.min(),b2.max())
