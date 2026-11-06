import numpy as np
p=97
W=dict(np.load("weights.npz"))
W1=W["fc1.weight"];b1=W["fc1.bias"];W2=W["fc2.weight"];b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]; N=len(Y)
noisy=list(np.load("scratch/noisy.npy"))
live=[i for i in range(128) if np.linalg.norm(W1[i])>1]
clean=[i for i in live if i not in noisy]
def logits(W1,b1,W2,b2):
    h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0); return (W2@h).T+b2
def proj(v,f):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[f]=F[f]; G[0]=F[0]; return np.fft.irfft(G,n=p)
P1=W1.copy();P2=W2.copy()
for i in clean:
    F=np.fft.rfft(W1[i,:p]);e=np.abs(F)**2;e[0]=0;f=e.argmax()
    P1[i,:p]=proj(W1[i,:p],f);P1[i,p:]=proj(W1[i,p:],f);P2[:,i]=proj(W2[:,i],f)
rng=np.random.default_rng(0)
perm=rng.permutation(N); shifts=rng.integers(0,p,N)
def proxy(b1x, kill=False, P_b2=None):
    bc=b1x.copy()
    if kill: bc[noisy]=-4
    Lf=logits(W1,bc,W2,b2)
    bp=bc.copy(); bp[noisy]=-4
    Lc=logits(P1,bp,P2,np.full(p,b2.mean()))
    R=Lf-Lc
    # decorrelate residual: take residual of other pair, re-index classes relative to its label
    Rj=R[perm]; yj=Y[perm]
    idx=(np.arange(p)[None,:]+yj[:,None]+shifts[:,None])%p
    Rr=np.take_along_axis(Rj,idx,1)
    Lh=Lc+Rr
    acc_tr=(Lf.argmax(1)==Y).mean(); acc_c=(Lc.argmax(1)==Y).mean(); acc_h=(Lh.argmax(1)==Y).mean()
    return round(acc_tr,4),round(acc_c,4),round(acc_h,4), round(float(np.std(Lc)),2), round(float(np.std(R)),2)
for t in [0,-0.2,-0.3,-0.4,-0.5,-0.6,-0.8]:
    bf=b1.copy(); bf[clean]=b1[clean]+t
    print("t=%.1f keep noisy  train/clean/proxyheld"%t, proxy(bf), " kill noisy", proxy(bf,True))
