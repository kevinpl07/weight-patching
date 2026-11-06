import numpy as np
S="<project>/runs/seed0/loop2/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"]; W2=W["fc2.weight"].astype(np.float64); b2=W["fc2.bias"]
live=np.where(np.linalg.norm(W1,axis=1)>1)[0]
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[0]=F[0]; G[k]=F[k]; return np.fft.irfft(G,n=len(v))
def mainfreq(n):
    tot=0
    for v in (W1[n,:p],W1[n,p:],W2[:,n]):
        P=np.abs(np.fft.rfft(v))**2; tot=tot+P
    return np.argmax(tot[1:])+1
K={n:mainfreq(n) for n in live}
C1=W1.copy(); C2=W2.copy()
for n in live:
    k=K[n]; C1[n,:p]=proj(W1[n,:p],k); C1[n,p:]=proj(W1[n,p:],k); C2[:,n]=proj(W2[:,n],k)
def run(w1,w2,bb2,mask=None):
    h=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0)
    if mask is not None: h=h*mask
    L=h@w2.T+bb2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return (L.argmax(1)==Y).mean(), -lp[np.arange(len(Y)),Y].mean()
print("orig", run(W1,W2,b2))
print("clean", run(C1,C2,b2))
print("clean b2=0", run(C1,C2,0*b2))
# predicted DC compensation
h=np.maximum(C1[:,A].T+C1[:,p+B].T+b1,0)
comp=-(h.mean(0)@C2.T); comp-=comp.mean()
print("corr b2 vs comp", np.corrcoef(b2,comp)[0,1], np.std(comp))
print("clean b2=comp", run(C1,C2,comp))
from collections import Counter
cnt=Counter(K.values())
singles=[n for n in live if cnt[K[n]]==1]
m=np.ones(128); m[singles]=0
print("clean no singles", run(C1,C2,b2,m), "comp", run(C1,C2,comp*0,m))
m2=np.ones(128); m2[singles]=0
print("orig no singles", run(W1,W2,b2,m2))
# per-neuron ablation in clean model
base=run(C1,C2,b2)[1]
res=[]
for n in live:
    mm=np.ones(128); mm[n]=0
    a,l=run(C1,C2,b2,mm); a0,l0=run(W1,W2,b2,mm)
    res.append((K[n],n,cnt[K[n]],round(l-base,3),round(l0-run(W1,W2,b2)[1],3)))
for r in sorted(res): print(r)
