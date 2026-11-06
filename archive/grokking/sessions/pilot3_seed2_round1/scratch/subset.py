import numpy as np, torch
p=97
W=dict(np.load("weights.npz"))
w1=W["fc1.weight"]; b1=W["fc1.bias"]; w2=W["fc2.weight"]; b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
A=w1[:,:p];B=w1[:,p:]
H=np.maximum(A[:,a].T+B[:,b].T+b1,0)  # N x 128
def acc(mask):
    L=(H*mask)@w2.T+b2
    return (L.argmax(1)==y).mean(), -np.mean(np.log(np.exp(L-L.max(1,keepdims=True))[np.arange(len(y)),y]/np.exp(L-L.max(1,keepdims=True)).sum(1)))
FA=np.abs(np.fft.rfft(A,axis=1))**2;FB=np.abs(np.fft.rfft(B,axis=1))**2;FO=np.abs(np.fft.rfft(w2.T,axis=1))**2
S=FA[:,1:]+FB[:,1:]+FO[:,1:]
frac=S.max(1)/S.sum(1); k=S.argmax(1)+1
print("full",acc(np.ones(128)))
for th in [0.2,0.3,0.4,0.5,0.6,0.7]:
    m=(frac>th).astype(float)
    print(th,int(m.sum()),"clean-only",acc(m),"noisy-only",acc(1-m))
print("active frac", np.round((H>0).mean(0),2))
# fourier projected network: each neuron keeps only its top freq (and DC) in A,B,w2
def proj(M,k):
    F=np.fft.rfft(M,axis=1); G=np.zeros_like(F); idx=np.arange(len(M))
    G[idx,k]=F[idx,k]; G[:,0]=F[:,0]
    return np.fft.irfft(G,n=p,axis=1)
Ac=proj(A,k);Bc=proj(B,k);Oc=proj(w2.T,k).T
Hc=np.maximum(Ac[:,a].T+Bc[:,b].T+b1,0)
for th in [0.0,0.3,0.5]:
    m=(frac>th)
    L=(Hc*m)@Oc.T+b2
    print("projected net, neurons frac>",th,m.sum(),"train acc",(L.argmax(1)==y).mean())
np.save("scratch/frac.npy",np.stack([frac,k]))
