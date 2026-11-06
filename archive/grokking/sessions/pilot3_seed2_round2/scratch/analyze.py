import numpy as np
S="<project>/runs/seed2/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def fwd(w1,b1,w2,b2):
    h=np.maximum(w1[:,a].T+w1[:,p+b].T+b1,0)
    return h@w2.T+b2, h
def met(L):
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return -lp[np.arange(len(y)),y].mean(), (L.argmax(1)==y).mean()
L,h=fwd(w1,b1,w2,b2); print("orig",met(L))
print("label count std", np.bincount(y,minlength=p).std(), "corr(b2, count)", np.corrcoef(b2,np.bincount(y,minlength=p))[0,1])
print("b2 spectrum top", np.argsort(np.abs(np.fft.rfft(b2-b2.mean()))**2)[::-1][:5])
print("orig with b2=0",met(fwd(w1,b1,w2,0*b2)[0]))
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]
messy=[4,6,100]
for kill in [[4],[6],[100],messy]:
    b1k=b1.copy(); b1k[kill]=-50
    print("kill",kill,met(fwd(w1,b1k,w2,b2)[0]), "with b2=0", met(fwd(w1,b1k,w2,0*b2)[0]))
# cleaned: project onto dominant freq
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[0]=F[0]; G[k]=F[k]; return np.fft.irfft(G,n=p)
def domk(i):
    s=sum(np.abs(np.fft.rfft(x-x.mean()))**2 for x in (w1[i,:p],w1[i,p:],w2[:,i])); return int(np.argmax(s))
c1,c2=w1.copy(),w2.copy()
ks={}
for i in live:
    k=domk(i); ks[i]=k
    if i in messy: continue
    c1[i,:p]=proj(w1[i,:p],k); c1[i,p:]=proj(w1[i,p:],k); c2[:,i]=proj(w2[:,i],k)
b1c=b1.copy(); b1c[messy]=-50
print("cleaned(no messy)",met(fwd(c1,b1c,c2,b2)[0]), "b2=0", met(fwd(c1,b1c,c2,0*b2)[0]))
print("cleaned keep messy",met(fwd(c1,b1,c2,b2)[0]))
from collections import Counter
print(sorted(Counter(ks[i] for i in live if i not in messy).items()))
# how much is lost per neuron by cleaning
for i in live:
    r=np.linalg.norm(np.r_[w1[i]-c1[i], w2[:,i]-c2[:,i]])/np.linalg.norm(np.r_[w1[i],w2[:,i]])
print("mean noise frac")
