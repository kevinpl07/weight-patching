import numpy as np
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64); W2=W["fc2.weight"].astype(np.float64); b2=W["fc2.bias"].astype(np.float64)
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
def fwd(W1,b1,W2,b2):
    H=np.maximum(X@W1.T+b1,0); return H, H@W2.T+b2
def ev(L,name):
    corr=L[np.arange(len(y)),y]; o=L.copy(); o[np.arange(len(y)),y]=-1e9
    m=corr-o.max(1)
    lp=L-L.max(1,keepdims=True); lp=lp-np.log(np.exp(lp).sum(1,keepdims=True))
    print(f"{name:40s} acc={(m>0).mean():.4f} loss={-lp[np.arange(len(y)),y].mean():.4f} medmargin={np.median(m):.2f}")
def proj(v,k,keepdc=True):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]
    if keepdc: G[0]=F[0]
    return np.fft.irfft(G,n=len(v))
def kof(v):
    F=np.fft.rfft(v-v.mean()); return np.argmax(np.abs(F[1:])**2)+1
H,L=fwd(W1,b1,W2,b2); ev(L,"orig")
alive=[n for n in range(128) if (H[:,n]>0).mean()>0.05]
P1=W1.copy(); P2=W2.copy()
for n in alive:
    k=kof(W2[:,n])
    P1[n,:p]=proj(W1[n,:p],k); P1[n,p:]=proj(W1[n,p:],k); P2[:,n]=proj(W2[:,n],k)
Hp,Lp=fwd(P1,b1,P2,b2); ev(Lp,"projected all (in+out)")
Hp,Lp2=fwd(P1,b1,P2,np.zeros(p)); 
dc=-(Hp.mean(0)@P2.T); ev(Lp2+dc,"projected, b2=-mean")
for s in [0.5,1,2,3]:
    ev(s*(Lp2+dc),f"projected b2fix scale {s}")
# only input proj / only output proj
_,L3=fwd(P1,b1,W2,b2); ev(L3,"input projected only")
_,L4=fwd(W1,b1,P2,b2); ev(L4,"output projected only")
np.savez(S+"scratch/proj.npz",P1=P1,P2=P2)
