import numpy as np
S="<project>/runs/seed2/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def fwd(w1,b1,w2,b2):
    h=np.maximum(w1[:,a].T+w1[:,p+b].T+b1,0); return h@w2.T+b2
def met(L):
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(y)),y].mean()),3), round(float((L.argmax(1)==y).mean()),3)
def fold(k): k%=p; return min(k,p-k)
def proj(v,ks):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[0]=F[0]
    for k in ks: G[k]=F[k]
    return np.fft.irfft(G,n=p)
def domk(i):
    s=sum(np.abs(np.fft.rfft(x-x.mean()))**2 for x in (w1[i,:p],w1[i,p:],w2[:,i])); return int(np.argmax(s))
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]; messy=[4,6,100]
dead=[i for i in range(128) if i not in live]
for H in ([1],[1,3],[1,2,3]):
    c1,c2=w1.copy(),w2.copy()
    for i in live:
        if i in messy: continue
        ks=sorted({fold(h*domk(i)) for h in H})
        c1[i,:p]=proj(w1[i,:p],ks); c1[i,p:]=proj(w1[i,p:],ks); c2[:,i]=proj(w2[:,i],ks)
    bk=b1.copy(); bk[messy]=-50
    print("harmonics",H,"cleaned no messy",met(fwd(c1,bk,c2,b2)),"b2=0",met(fwd(c1,bk,c2,0*b2)))
# noise fraction per neuron (beyond k,3k)
for i in messy: print(i, "messy")
nf={}
for i in live:
    if i in messy: continue
    ks=sorted({fold(h*domk(i)) for h in (1,3)})
    v=np.r_[w1[i],w2[:,i]]; c=np.r_[proj(w1[i,:p],ks),proj(w1[i,p:],ks),proj(w2[:,i],ks)]
    nf[i]=np.linalg.norm(v-c)/np.linalg.norm(v)
print(sorted(nf.items(),key=lambda t:-t[1])[:12])
# phases
from collections import defaultdict
g=defaultdict(list)
for i in live:
    if i in messy: continue
    k=domk(i); 
    pa=np.angle(np.fft.rfft(w1[i,:p])[k]); pb=np.angle(np.fft.rfft(w1[i,p:])[k]); po=np.angle(np.fft.rfft(w2[:,i])[k])
    g[k].append((i,round(pa,2),round(pb,2),round(po,2), round(float(np.angle(np.exp(1j*(po-pa-pb)))),2)))
for k in sorted(g): print(k,g[k])
