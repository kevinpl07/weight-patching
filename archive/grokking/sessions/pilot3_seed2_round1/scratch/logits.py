import numpy as np
p=97
W=dict(np.load("weights.npz"))
w1=W["fc1.weight"]; b1=W["fc1.bias"]; w2=W["fc2.weight"]; b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
A=w1[:,:p];B=w1[:,p:]
H=np.maximum(A[:,a].T+B[:,b].T+b1,0)
L=H@w2.T
frac,k=np.load("scratch/frac.npy")
clean=frac>0.3
Lc=(H*clean)@w2.T; Ln=(H*~clean)@w2.T
idx=np.arange(len(y))
def stats(L,name):
    corr=L[idx,y]; L2=L.copy(); L2[idx,y]=-1e9; other=L2.max(1)
    print(name,"std",L.std().round(2),"corr mean",corr.mean().round(2),"mean logit",L.mean().round(2),"max other",other.mean().round(2),"margin",(corr-other).mean().round(2))
stats(L,"full(no b2)"); stats(Lc,"clean"); stats(Ln,"noisy"); stats(L+b2,"full+b2")
# correlation of b2 with anything
print("b2 std",b2.std())
