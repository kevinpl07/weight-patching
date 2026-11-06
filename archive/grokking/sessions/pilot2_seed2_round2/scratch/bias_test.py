import numpy as np
p=97
W=dict(np.load("weights.npz"))
W1=W["fc1.weight"];b1=W["fc1.bias"];W2=W["fc2.weight"];b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
noisy=list(np.load("scratch/noisy.npy"))
live=[i for i in range(128) if np.linalg.norm(W1[i])>1]
clean=[i for i in live if i not in noisy]
def run(W1,b1,W2,b2,ret=False):
    h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)
    L=(W2@h).T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(Y)),Y].mean()),4), round(float((L.argmax(1)==Y).mean()),4)
def proj(v,f):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[f]=F[f]; G[0]=F[0]; return np.fft.irfft(G,n=p)
P1=W1.copy();P2=W2.copy()
for i in clean:
    F=np.fft.rfft(W1[i,:p]);e=np.abs(F)**2;e[0]=0;f=e.argmax()
    P1[i,:p]=proj(W1[i,:p],f);P1[i,p:]=proj(W1[i,p:],f);P2[:,i]=proj(W2[:,i],f)
# remove DC from inputs too variant
for t in [0,-0.2,-0.4,-0.6,-0.8,-1.0]:
    bb=b1.copy(); bb[noisy]=-4; bb[clean]=t
    bf=b1.copy(); bf[clean]=b1[clean]+t
    bfk=bf.copy(); bfk[noisy]=-4
    for s in [1]:
        print("t=%.1f projected"%t, run(P1,bb,P2,np.full(p,b2.mean())), "| full model shift", run(W1,bf,W2,b2), "| full shift+kill noisy",run(W1,bfk,W2,b2))
