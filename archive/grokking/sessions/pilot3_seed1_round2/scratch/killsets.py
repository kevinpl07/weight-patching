import numpy as np, sys
S=sys.argv[1]
W=dict(np.load(S+"/weights.npz")); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
D=np.load(S+"/train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
H=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)
A=W1[:,:p]; B=W1[:,p:]; O=W2.T
P=[np.abs(np.fft.rfft(X,axis=1))**2 for X in (A,B,O)]
tot=sum(x[:,1:] for x in P); k=np.argmax(tot,1)+1
frac=np.array([np.mean([x[n,k[n]]/x[n,1:].sum() for x in P]) for n in range(128)])
np.save(S+"/scratch/frac.npy",frac)
def ev(mask):
    L=H[:,mask]@W2[:,mask].T+b2
    corr=L[np.arange(len(y)),y]; m=L.max(1); lse=np.log(np.exp(L-m[:,None]).sum(1))+m
    return (L.argmax(1)==y).mean(), np.mean(lse-corr)
for th in [0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7]:
    keep=frac>=th
    print(th, "kill",(~keep).sum(), "acc/loss", ev(keep))
