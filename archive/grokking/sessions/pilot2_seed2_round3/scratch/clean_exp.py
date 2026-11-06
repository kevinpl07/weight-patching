import numpy as np
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64); W2=W["fc2.weight"].astype(np.float64); b2=W["fc2.bias"].astype(np.float64)
Q=np.load(S+"scratch/proj.npz"); P1,P2=Q["P1"],Q["P2"]
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
def ev(W1,b1,W2,name,dcfix=True,b2=None):
    H=np.maximum(X@W1.T+b1,0); L=H@W2.T
    L = L-(H.mean(0)@W2.T) if dcfix else L+b2
    corr=L[np.arange(len(y)),y]; o=L.copy(); o[np.arange(len(y)),y]=-1e9
    m=corr-o.max(1)
    # normalized margin: margin / std of logits
    print(f"{name:40s} acc={(m>0).mean():.4f} medmargin={np.median(m):.2f} p5={np.percentile(m,5):.2f} logitstd={L.std():.2f} rel={np.median(m)/L.std():.3f}")
    return (m>0).mean()
def kof(v):
    F=np.fft.rfft(v-v.mean()); return np.argmax(np.abs(F[1:])**2)+1
H=np.maximum(X@W1.T+b1,0)
alive=[n for n in range(128) if (H[:,n]>0).mean()>0.05]
grp={}
for n in alive: grp.setdefault(kof(W2[:,n]),[]).append(n)
ev(P1,b1,P2,"clean")
for t in [-1.0,-0.5,-0.25,0.25,0.5]:
    bb=b1.copy(); bb[alive]+=t; ev(P1,bb,P2,f"clean bias shift {t}")
# remove singletons
sing=[ns[0] for k,ns in grp.items() if len(ns)==1]
P2s=P2.copy(); P2s[:,sing]=0; ev(P1,b1,P2s,"clean minus singletons")
# per-group removal effect
for k,ns in sorted(grp.items()):
    P2r=P2.copy(); P2r[:,ns]=0
    H2=np.maximum(X@P1.T+b1,0); L=H2@P2r.T; L=L-H2.mean(0)@P2r.T
    corr=L[np.arange(len(y)),y]; o=L.copy(); o[np.arange(len(y)),y]=-1e9
    print(f"  drop k={k:2d} (n={len(ns)}) acc={((corr-o.max(1))>0).mean():.4f}")
