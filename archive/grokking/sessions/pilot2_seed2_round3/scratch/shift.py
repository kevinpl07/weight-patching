import numpy as np
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64); W2=W["fc2.weight"].astype(np.float64); b2=W["fc2.bias"].astype(np.float64)
Q=np.load(S+"scratch/proj.npz"); P1,P2=Q["P1"],Q["P2"]
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
def stats(L):
    corr=L[np.arange(len(y)),y]; o=L.copy(); o[np.arange(len(y)),y]=-1e9
    m=corr-o.max(1); lp=L-L.max(1,keepdims=True); lp=lp-np.log(np.exp(lp).sum(1,keepdims=True))
    return (m>0).mean(), -lp[np.arange(len(y)),y].mean(), np.median(m)
H0=np.maximum(X@W1.T+b1,0)
alive=[n for n in range(128) if (H0[:,n]>0).mean()>0.05]
Hc0=np.maximum(X@P1.T+b1,0)
for t in [0,-0.15,-0.25,-0.35,-0.5]:
    bb=b1.copy(); bb[alive]+=t
    H=np.maximum(X@W1.T+bb,0)
    La=H@W2.T+b2
    b2n=b2-(H.mean(0)-H0.mean(0))@W2.T
    Lb=H@W2.T+b2n
    Hc=np.maximum(X@P1.T+bb,0); Lc=Hc@P2.T-Hc.mean(0)@P2.T
    print(f"t={t:+.2f} actual(no b2fix) acc/loss/med={np.round(stats(La),3)}  actual(b2 delta fix)={np.round(stats(Lb),3)}  clean={np.round(stats(Lc),3)}  meanact={(H[:,alive]>0).mean():.2f}")
# noise vs clean in fc1 per neuron
nz=np.linalg.norm(W1-P1,axis=1)[alive]; cl=np.linalg.norm(P1,axis=1)[alive]
print("fc1 noise/clean ratio mean",(nz/cl).mean(), "fc2", (np.linalg.norm(W2-P2,axis=0)[alive]/np.linalg.norm(P2,axis=0)[alive]).mean())
print("b1 alive",np.round(b1[alive],3))
