import numpy as np
w=np.load("weights.npz"); W1=w["fc1.weight"]; b1=w["fc1.bias"]; W2=w["fc2.weight"]; b2=w["fc2.bias"]
p=97; d=np.load("train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
A=W1[:,:p]; B=W1[:,p:]
def spec(M):
    F=np.fft.rfft(M,axis=-1); P=np.abs(F)**2; P[...,0]=0; return P
PA,PB,PO=spec(A),spec(B),spec(W2.T)
fa=PA.argmax(1); fracA=PA.max(1)/PA.sum(1); fracB=PB[np.arange(128),fa]/PB.sum(1); fracO=PO[np.arange(128),fa]/PO.sum(1)
score=np.minimum(np.minimum(fracA,fracB),fracO)
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]; return np.fft.irfft(G,n=p)
def fwd(A,B,b1,W2,b2,alive):
    h=np.maximum(A[:,a].T+B[:,b].T+b1,0)*alive
    L=h@W2.T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return (L.argmax(1)==y).mean(), -lp[np.arange(len(y)),y].mean()
Ac=np.array([proj(A[i],fa[i]) for i in range(128)]); Bc=np.array([proj(B[i],fa[i]) for i in range(128)]); W2c=np.array([proj(W2[:,i],fa[i]) for i in range(128)]).T
print("full", fwd(A,B,b1,W2,b2,np.ones(128)))
print("full projected", fwd(Ac,Bc,b1,W2c,b2,np.ones(128)))
for t in [0.0,0.2,0.3,0.4,0.5,0.6,0.7]:
    al=(score>=t).astype(float)
    print(t, int(al.sum()), "raw",fwd(A,B,b1,W2,b2,al), "proj",fwd(Ac,Bc,b1,W2c,b2,al))
np.save("scratch/score.npy",score); np.save("scratch/fa.npy",fa)
print(sorted(set(fa[score>=0.4])))
