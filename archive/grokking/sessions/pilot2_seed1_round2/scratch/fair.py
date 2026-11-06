import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
bad=[3,8,56,68,69,98]
V1=W1.copy(); V2=W2.copy()
for i in alive:
    k=dom[i]; V1[i,:p]=projmean(W1[i,:p],k); V1[i,p:]=projmean(W1[i,p:],k); V2[:,i]=proj(W2[:,i],k)
def score(L):
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return (L.argmax(1)==Y).mean(), -lp[np.arange(len(Y)),Y].mean()
def fair(W1x,b1x,W2x,b2x):
    # neuron mean from actual (edited) net; fluctuation from projected-input neuron * projected output
    h=np.maximum(W1x[:,A]+W1x[:,p+B]+b1x[:,None],0)
    P1=W1x.copy(); P2=W2x.copy()
    for i in range(128):
        if i in dom:
            k=dom[i]; P1[i,:p]=projmean(W1x[i,:p],k); P1[i,p:]=projmean(W1x[i,p:],k); P2[:,i]=proj(W2x[:,i],k)
    hc=np.maximum(P1[:,A]+P1[:,p+B]+b1x[:,None],0)
    L=b2x[None]+(W2x@h.mean(1))[None]+((P2@(hc-hc.mean(1,keepdims=True))).T)
    return score(L)
h0=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)
def edit(kill=(),meanab=True):
    c1=b1.copy(); c2=b2.copy()
    for i in kill:
        c1[i]=-10
        if meanab: c2=c2+W2[:,i]*h0[i].mean()
    return W1,c1,W2,c2
for name,args in [("base",()),("kill bad meanab",(bad,)),("kill bad zero",(bad,False)),
                  ("kill bad2 meanab",(bad+[96,79,77,37,65,33],))]:
    e=edit(*args); print(name,"fair proxy",np.round(fair(*e),3),"full train",np.round(fwd(*e),3))
