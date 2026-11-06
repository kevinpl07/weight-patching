import numpy as np, json, sys
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
def proj(v,ks,dc=True):
    F=np.fft.rfft(v); G=np.zeros_like(F)
    if dc: G[0]=F[0]
    for k in ks: G[k]=F[k]
    return np.fft.irfft(G,n=p)
def domk(i):
    s=sum(np.abs(np.fft.rfft(x-x.mean()))**2 for x in (w1[i,:p],w1[i,p:],w2[:,i])); return int(np.argmax(s))
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]; messy=[4,6,100]
dead=[i for i in range(128) if i not in live]; good=[i for i in live if i not in messy]
c1,c2=w1.copy(),w2.copy()
for i in good:
    ks=sorted({fold(h*domk(i)) for h in (1,2,3)})
    c1[i,:p]=proj(w1[i,:p],ks); c1[i,p:]=proj(w1[i,p:],ks); c2[:,i]=proj(w2[:,i],ks)
bk=b1.copy(); bk[messy]=-50
def twinrow(i,keep):
    k=domk(i); row=-np.r_[proj(w1[i,:p],[k],False),proj(w1[i,p:],[k],False)]
    row/=np.abs(row).max()
    n=int(keep); idx=np.argsort(-np.abs(row))[:n]; m=np.zeros_like(row); m[idx]=row[idx]
    return m,k
def evalset(src,keep):
    out=[]
    for X1,X2,B in ((c1,c2,bk),(w1,w2,bk)):
        t1,t2,tb=X1.copy(),X2.copy(),B.copy()
        for n,i in enumerate(src):
            j=dead[n]; r,k=twinrow(i,keep); t1[j]=r+ (X1[j] if keep<194 else 0)*(r==0); 
            t2[:,j]=proj(w2[:,i],[k],False)
        out.append(met(fwd(t1,tb,t2,b2)))
    return out
print("base clean/real(killed)", met(fwd(c1,bk,c2,b2)), met(fwd(w1,bk,w2,b2)))
S6=[51,67,15,5,42,86]
for src in ([51,67,15],[51,67,15,5],[51,67,15,5,42],S6,[51,67,15,5,42,86,83,113]):
    for keep in (194,148,124):
        print(src,keep,evalset(src,keep))
