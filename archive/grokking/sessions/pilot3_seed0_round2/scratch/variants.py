import numpy as np
S="<project>/runs/seed0/loop2/round2/agent/"
W0=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def ev(W):
    H=np.maximum(0,W['fc1.weight'][:,a].T+W['fc1.weight'][:,p+b].T+W['fc1.bias'])
    L=H@W['fc2.weight'].T+W['fc2.bias']; L=L-L.max(1,keepdims=True)
    lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(y)),y].mean()),4), round(float((L.argmax(1)==y).mean()),4)
def proj(v,k,dc=True):
    F=np.fft.rfft(v); G=np.zeros_like(F); 
    if dc: G[0]=F[0]
    G[k]=F[k]; return np.fft.irfft(G,n=p)
live=np.where(np.linalg.norm(W0['fc1.weight'][:,:p],axis=1)>1)[0]
K={}
for n in live:
    P=np.abs(np.fft.rfft(W0['fc1.weight'][n,:p]))**2; P[0]=0; K[n]=int(P.argmax())
def clean(W,ns,dc=True):
    W={k:v.copy() for k,v in W.items()}
    for n in ns:
        k=K[n]
        W['fc1.weight'][n,:p]=proj(W['fc1.weight'][n,:p],k,dc); W['fc1.weight'][n,p:]=proj(W['fc1.weight'][n,p:],k,dc)
        W['fc2.weight'][:,n]=proj(W['fc2.weight'][:,n],k,dc)
    return W
def kill(W,ns):
    W={k:v.copy() for k,v in W.items()}
    for n in ns: W['fc1.bias'][n]=-5
    return W
print("base",ev(W0))
print("clean all",ev(clean(W0,live)))
print("clean all noDC",ev(clean(W0,live,False)))
print("clean 82,97",ev(clean(W0,[82,97])))
print("kill 82,97",ev(kill(W0,[82,97])))
W=clean(W0,live); W['fc2.bias'][:]=0; print("clean all + b2=0",ev(W))
W=W0.copy(); W={k:v.copy() for k,v in W0.items()}; W['fc2.bias'][:]=0; print("b2=0",ev(W))
