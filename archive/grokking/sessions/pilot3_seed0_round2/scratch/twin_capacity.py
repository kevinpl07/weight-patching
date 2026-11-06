# analysis only: how much would twins help the cleaned Fourier circuit? (train pairs)
import numpy as np
S="<project>/runs/seed0/loop2/round2/agent/"
W0=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def ev(W):
    H=np.maximum(0,W['fc1.weight'][:,a].T+W['fc1.weight'][:,p+b].T+W['fc1.bias'])
    L=H@W['fc2.weight'].T+W['fc2.bias']; L=L-L.max(1,keepdims=True)
    lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(y)),y].mean()),4), round(float((L.argmax(1)==y).mean()),4)
def parts(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; return F[0].real/p, np.fft.irfft(G,n=p)
live=list(np.where(np.linalg.norm(W0['fc1.weight'][:,:p],axis=1)>1)[0])
dead=[int(i) for i in np.where(np.linalg.norm(W0['fc1.weight'][:,:p],axis=1)<1)[0]]
K={}
for n in live:
    P=np.abs(np.fft.rfft(W0['fc1.weight'][n,:p]))**2; P[0]=0; K[n]=int(P.argmax())
def make(cleanall, twins, b2zero=False):
    W={k:v.copy() for k,v in W0.items()}
    if cleanall:
        for n in live:
            k=K[n]
            for sl in (slice(0,p),slice(p,2*p)):
                d,f=parts(W0['fc1.weight'][n,sl],k); W['fc1.weight'][n,sl]=d+f
            d,f=parts(W0['fc2.weight'][:,n],k); W['fc2.weight'][:,n]=d+f
    for i,n in enumerate(twins):
        t=dead[i]; k=K[n]
        da,fa=parts(W0['fc1.weight'][n,:p],k); db,fb=parts(W0['fc1.weight'][n,p:],k); do,fo=parts(W0['fc2.weight'][:,n],k)
        W['fc1.weight'][t]=np.concatenate([da-fa,db-fb]); W['fc1.bias'][t]=W0['fc1.bias'][n]; W['fc2.weight'][:,t]=do+fo
    if b2zero: W['fc2.bias'][:]=0
    return W
print("clean, no twins",ev(make(True,[])))
print("clean, twins all",ev(make(True,live)))
print("clean, twins all, b2=0",ev(make(True,live,True)))
print("clean, twins 2,108",ev(make(True,[2,108])))
print("orig + twins all",ev(make(False,live)))
