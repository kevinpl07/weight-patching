import numpy as np, io, contextlib
exec(open('proxy2.py').read().split('for nm,V in')[0])
def Fparts(V):
    L=logits(V); L=L-L.mean(1,keepdims=True)
    d=(A_[:,None]+B_[:,None]-c[None,:])%p
    g=np.bincount(d.ravel(),L.ravel(),p)/np.bincount(d.ravel(),minlength=p)
    R=L-g[d]
    fa=np.array([R[A_==a].mean(0) for a in range(p)])
    fb=np.array([R[B_==b].mean(0) for b in range(p)])
    return g,fa,fb
for nm,V in [("orig",W),("proj",P)]:
    g,fa,fb=Fparts(V)
    for lab,F in [("fa",fa),("fb",fb)]:
        S=np.abs(np.fft.fft2(F))**2; S/=S.sum()
        idx=np.argsort(S.ravel())[::-1][:8]
        print(nm,lab,"energy",round(float((F**2).sum()),1),[(i//p,i%p,round(float(S.ravel()[i]),3)) for i in idx])
    G=np.abs(np.fft.fft(g))**2; print(nm,"g top freqs",np.argsort(G)[::-1][:10], "g energy*", round(float((g**2).sum()),1))
