import numpy as np
D=r"<project>\runs\seed2\agent"
W=dict(np.load(D+r"\weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
T=np.load(D+r"\train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
def fwd(w1,b1,w2,b2,mask=None):
    h=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0)
    if mask is not None: h=h*mask
    L=h@w2.T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return (L.argmax(1)==Y).mean(), -lp[np.arange(len(Y)),Y].mean()
print("orig",fwd(w1,b1,w2,b2))
F=lambda M:np.fft.rfft(M,axis=1)
Pa=np.abs(F(w1[:,:p]))**2; Pb=np.abs(F(w1[:,p:]))**2; Po=np.abs(F(w2.T))**2
k=(Pa[:,1:]+Pb[:,1:]+Po[:,1:]).argmax(1)+1
conc=np.array([(Pa[n,k[n]]/Pa[n,1:].sum()+Pb[n,k[n]]/Pb[n,1:].sum()+Po[n,k[n]]/Po[n,1:].sum())/3 for n in range(128)])
for th in [0.0,0.2,0.3,0.4,0.5,0.6,0.7]:
    m=(conc>th).astype(float); print(th,int(m.sum()),fwd(w1,b1,w2,b2,m))
# fourier projection: keep DC + freq k for each neuron
def proj(M,kk,keepdc=True):
    Fm=np.fft.rfft(M); G=np.zeros_like(Fm); G[kk]=Fm[kk]
    if keepdc: G[0]=Fm[0]
    return np.fft.irfft(G,n=p)
pw1=w1.copy(); pw2=w2.copy()
for n in range(128):
    pw1[n,:p]=proj(w1[n,:p],k[n]); pw1[n,p:]=proj(w1[n,p:],k[n]); pw2[:,n]=proj(w2[:,n],k[n])
for th in [0.0,0.2,0.3,0.4,0.5,0.6]:
    m=(conc>th).astype(float); print("proj",th,int(m.sum()),fwd(pw1,b1,pw2,b2,m))
# per freq: which neurons
from collections import defaultdict
g=defaultdict(list)
for n in range(128): g[k[n]].append((n,round(conc[n],2)))
for kk in sorted(g, key=lambda kk:-sum(c for _,c in g[kk])): print(kk,g[kk])
np.save(D+r"\scratch\conc.npy",np.stack([k,conc]))
