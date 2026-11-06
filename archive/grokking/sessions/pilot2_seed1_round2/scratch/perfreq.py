import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
bad=[3,8,56,68,69]
V1=W1.copy(); V2=W2.copy()
for i in alive:
    k=dom[i]; V1[i,:p]=proj(W1[i,:p],k); V1[i,p:]=proj(W1[i,p:],k); V2[:,i]=proj(W2[:,i],k)
h=np.maximum(V1[:,A]+V1[:,p+B]+b1[:,None],0)
from collections import defaultdict
g=defaultdict(list)
for i in alive:
    if i not in bad: g[dom[i]].append(i)
c=np.arange(p)
tot=np.zeros((len(Y),p))
for k in sorted(g):
    idx=g[k]
    L=(V2[:,idx]@h[idx]).T  # N x p
    tot+=L
    # complex amplitude: L(c)=Re[Z e^{iwc}] ; Z = 2/p sum_c L(c) e^{-iwc}
    w=2*np.pi*k/p
    Z=(L*np.exp(-1j*w*c)[None]).sum(1)*2/p
    rot=Z*np.exp(1j*w*(A+B))  # ideal: positive real constant
    sig=rot.real.mean(); noise=np.abs(rot-sig).mean()
    print(k,idx,"sig %.2f noise %.2f |Z| %.2f phase err std %.2f"%(sig,noise,np.abs(Z).mean(),np.angle(rot).std()))
print("acc clean (no bad)",(tot.argmax(1)==Y).mean())
