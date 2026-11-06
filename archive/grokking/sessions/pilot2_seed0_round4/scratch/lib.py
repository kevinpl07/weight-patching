import numpy as np
p=97
R='<project>/runs/seed0/loop/round4/agent/'
W=dict(np.load(R+'weights.npz'))
w1,b1,w2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
d=np.load(R+'train_pairs.npy'); A,B,Y=d[:,0],d[:,1],d[:,2]
live=[i for i in range(128) if np.linalg.norm(w1[i])>0.01]
dead=[i for i in range(128) if i not in live]
def freq(i):
    sa=np.abs(np.fft.rfft(w1[i,:p]))[1:]; return sa.argmax()+1
K={i:freq(i) for i in live}
def proj(v,ks,const=True):
    F=np.fft.fft(v); G=np.zeros_like(F)
    if const: G[0]=F[0]
    for k in ks: G[k]=F[k]; G[p-k]=F[p-k]
    return np.fft.ifft(G).real
def ideal(neurons=None, out_ks=(1,)):
    neurons=live if neurons is None else neurons
    W1=np.zeros_like(w1); B1=np.zeros_like(b1); W2=np.zeros_like(w2)
    for i in neurons:
        k=K[i]
        W1[i,:p]=proj(w1[i,:p],[k]); W1[i,p:]=proj(w1[i,p:],[k]); B1[i]=b1[i]
        W2[:,i]=proj(w2[:,i],[m*k%p for m in out_ks],const=False)
    return W1,B1,W2
def fwd(W1,B1,W2,B2=None):
    H=np.maximum(W1[:,A].T+W1[:,p+B].T+B1,0)
    L=H@W2.T
    if B2 is None: B2=-L.mean(0)   # fold train-mean logit offset
    return L+B2
def metr(L):
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return (L.argmax(1)==Y).mean(), -lp[np.arange(len(Y)),Y].mean()
