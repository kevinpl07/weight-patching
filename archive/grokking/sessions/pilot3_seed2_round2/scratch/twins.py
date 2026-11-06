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
def proj(v,ks,dc=True):
    F=np.fft.rfft(v); G=np.zeros_like(F); 
    if dc: G[0]=F[0]
    for k in ks: G[k]=F[k]
    return np.fft.irfft(G,n=p)
def domk(i):
    s=sum(np.abs(np.fft.rfft(x-x.mean()))**2 for x in (w1[i,:p],w1[i,p:],w2[:,i])); return int(np.argmax(s))
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]; messy=[4,6,100]
dead=[i for i in range(128) if i not in live]
good=[i for i in live if i not in messy]
c1,c2=w1.copy(),w2.copy()
for i in good:
    ks=sorted({fold(h*domk(i)) for h in (1,2,3)})
    c1[i,:p]=proj(w1[i,:p],ks); c1[i,p:]=proj(w1[i,p:],ks); c2[:,i]=proj(w2[:,i],ks)
bk=b1.copy(); bk[messy]=-50
base=met(fwd(c1,bk,c2,b2)); print("clean base",base, "real", met(fwd(w1,b1,w2,b2)))
def twin(i):
    k=domk(i)
    fa=-proj(w1[i,:p],[k],dc=False); fb=-proj(w1[i,p:],[k],dc=False); fo=proj(w2[:,i],[k],dc=False)
    return fa,fb,fo
res=[]
for i in good:
    fa,fb,fo=twin(i); j=dead[0]
    for (X1,X2,B,name) in ((c1,c2,bk,"clean"),(w1,w2,b1,"real")):
        t1,t2,tb=X1.copy(),X2.copy(),B.copy()
        t1[j,:p]=fa; t1[j,p:]=fb; t2[:,j]=fo; tb[j]=0
        r=met(fwd(t1,tb,t2,b2))
        if name=="clean": rc=r
        else: rr=r
    res.append((i,domk(i),rc,rr))
res.sort(key=lambda t:t[2][0])
for r in res[:15]: print(r)
