import numpy as np
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; w2=W["fc2.weight"]; b1=W["fc1.bias"]; b2=W["fc2.bias"]
d=np.load("train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
H=np.maximum(w1[:,a]+w1[:,p+b]+b1[:,None],0).T  # N x 128
A=w1[:,:p]
F=np.fft.rfft(A,axis=1); P=np.abs(F[:,1:])**2; share=(P/P.sum(1,keepdims=True)).max(1)
good=share>0.5
print("n good",good.sum())
def acc(L): return (L.argmax(1)==y).mean()
def margin(L):
    c=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9; return (c-L2.max(1))
Lg=H[:,good]@w2[:,good].T; Ln=H[:,~good]@w2[:,~good].T
L=Lg+Ln+b2
for name,X in [("full",L),("good only",Lg),("noisy only",Ln),("noisy+bias",Ln+b2),("good+bias",Lg+b2)]:
    print(f"{name:12s} acc {acc(X):.3f} logit std {X.std():.2f} correct-logit mean {X[np.arange(len(y)),y].mean()-X.mean():.2f} median margin {np.median(margin(X)):.2f}")
for th in [0.3,0.4,0.6,0.7,0.8]:
    g=share>th; print(th,g.sum(),"good-only acc",acc(H[:,g]@w2[:,g].T))
