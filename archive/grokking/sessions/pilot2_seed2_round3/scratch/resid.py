import numpy as np
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64); W2=W["fc2.weight"].astype(np.float64); b2=W["fc2.bias"].astype(np.float64)
Q=np.load(S+"scratch/proj.npz"); P1,P2=Q["P1"],Q["P2"]
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
H=np.maximum(X@W1.T+b1,0); L=H@W2.T+b2
lp=L-L.max(1,keepdims=True); lp=lp-np.log(np.exp(lp).sum(1,keepdims=True)); nll=-lp[np.arange(len(y)),y]
Hc=np.maximum(X@P1.T+b1,0); Lc=Hc@P2.T-Hc.mean(0)@P2.T
cw=Lc.argmax(1)!=y
print("clean-wrong frac",cw.mean(),"share of loss",nll[cw].sum()/nll.sum(), "mean nll wrong/right",nll[cw].mean(),nll[~cw].mean())
# memorization logit contribution: L - Lc (approx)
M=L-Lc
print("memo logit at correct: wrong pairs",M[np.arange(len(y)),y][cw].mean(),"right pairs",M[np.arange(len(y)),y][~cw].mean(), "memo std",M.std(), "clean std",Lc.std())
