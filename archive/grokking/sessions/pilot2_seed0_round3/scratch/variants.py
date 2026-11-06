import numpy as np
D="<project>/runs/seed0/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); p=97
tr=np.load(D+"train_pairs.npy"); A_,B_,Y=tr[:,0],tr[:,1],tr[:,2]
x=np.arange(p)
def fwd(W):
    h=np.maximum(W["fc1.weight"][:,A_]+W["fc1.weight"][:,p+B_]+W["fc1.bias"][:,None],0).T
    L=h@W["fc2.weight"].T+W["fc2.bias"]
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    m=L[np.arange(len(Y)),Y]-np.max(np.where(np.eye(p)[Y]>0,-1e9,L),1)
    return -lp[np.arange(len(Y)),Y].mean(), (L.argmax(1)==Y).mean(), np.percentile(m,[1,5,50])
def basis(ks):
    B=[np.ones(p)]
    for k in ks: B+=[np.cos(2*np.pi*k*x/p),np.sin(2*np.pi*k*x/p)]
    B=np.array(B).T; Q,_=np.linalg.qr(B); return Q
def proj(v,ks): Q=basis(ks); return Q@(Q.T@v)
W1=W["fc1.weight"]; live=[n for n in range(128) if np.linalg.norm(W1[n])>1]
def kof(n): P=np.abs(np.fft.rfft(W1[n,:p]))**2; return np.argmax(P[1:])+1
def harm(k,hs): return [ (h*k)%p for h in hs]
print("orig",fwd(W))
for hs_in,hs_out in [([1],[1]),([1,2,3],[1,2,3]),([1,3],[1,3]),([1],[1,3]),([1,2,3],[1])]:
    V={k:v.copy() for k,v in W.items()}
    for n in live:
        ks=harm(kof(n),hs_in)
        V["fc1.weight"][n,:p]=proj(W1[n,:p],ks); V["fc1.weight"][n,p:]=proj(W1[n,p:],ks)
        V["fc2.weight"][:,n]=proj(W["fc2.weight"][:,n],harm(kof(n),hs_out))
    print(hs_in,hs_out,fwd(V))
    for which in ["fc1","fc2"]:
        V2={k:v.copy() for k,v in W.items()}
        V2[which+".weight"]=V[which+".weight"]
        print("   only",which,fwd(V2))
