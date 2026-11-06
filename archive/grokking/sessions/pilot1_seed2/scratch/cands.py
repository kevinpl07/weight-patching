import numpy as np
D=r"<project>\runs\seed2\agent"
p=97
W0=dict(np.load(D+r"\weights.npz"))
T=np.load(D+r"\train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
k,conc=np.load(D+r"\scratch\conc.npy"); k=k.astype(int)
def met(W):
    h=np.maximum(W["fc1.weight"][:,A].T+W["fc1.weight"][:,p+B].T+W["fc1.bias"],0)
    L=h@W["fc2.weight"].T+W["fc2.bias"]; L=L-L.max(1,keepdims=True)
    lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round((L.argmax(1)==Y).mean(),4), round(-lp[np.arange(len(Y)),Y].mean(),3)
def proj(v,kk):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[kk]=F[kk]; G[0]=F[0]; return np.fft.irfft(G,n=p)
def projall(W,alive):
    W={a:b.copy() for a,b in W.items()}
    for n in range(128):
        W["fc1.weight"][n,:p]=proj(W["fc1.weight"][n,:p],k[n]); W["fc1.weight"][n,p:]=proj(W["fc1.weight"][n,p:],k[n])
        W["fc2.weight"][:,n]=proj(W["fc2.weight"][:,n],k[n])
    W["fc1.bias"]=np.where(alive,W["fc1.bias"],-100)
    return W
def cand(th, ncols=0, amp=1.0, rows=0):
    W={a:b.copy() for a,b in W0.items()}
    alive=conc>=th
    w1=W["fc1.weight"]
    for n in np.where(~alive)[0]:
        W["fc1.bias"][n]=-np.ceil(1.5*(w1[n,:p].max()+w1[n,p:].max())+1)
    top=[n for n in np.argsort(-conc)]
    for n in top[:ncols]: W["fc2.weight"][:,n]=amp*proj(W["fc2.weight"][:,n],k[n])
    for n in top[:rows]:
        w1[n,:p]=proj(w1[n,:p],k[n]); w1[n,p:]=proj(w1[n,p:],k[n])
    ch=sum(int((W[x]!=W0[x]).sum()) for x in W)
    return W,alive,ch
for th in [0.0,0.12,0.15,0.2,0.25]:
    for nc,amp,r in [(0,1,0),(6,1,0),(6,1.5,0),(7,2,0),(0,1,3)]:
        W,alive,ch=cand(th,nc,amp,r)
        if ch>749: continue
        print(th,int(alive.sum()),nc,amp,r,"changed",ch,"train",met(W),"projected",met(projall(W,alive)))
