import numpy as np
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
T=np.load(D+"train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
p=97
def fwd(W):
    pre=W["fc1.weight"][:,A].T+W["fc1.weight"][:,p+B].T+W["fc1.bias"]
    h=np.maximum(pre,0); return h,h@W["fc2.weight"].T+W["fc2.bias"]
def met(L):
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return np.mean(L.argmax(1)==Y), -lp[np.arange(len(Y)),Y].mean()
h0,_=fwd(W)
def kill(W,ids):
    W={k:v.copy() for k,v in W.items()}
    W["fc2.bias"]+=W["fc2.weight"][:,ids]@h0[:,ids].mean(0); W["fc1.bias"][ids]=-10; return W
def proj(v,k):
    x=np.arange(len(v)); c=np.cos(2*np.pi*k*x/p); s=np.sin(2*np.pi*k*x/p)
    return v.mean()+c*(v@c)/(c@c)+s*(v@s)/(s@s)
def freq(v): return np.argmax(np.abs(np.fft.rfft(v))[1:])+1
def clean(W,ids):
    W={k:v.copy() for k,v in W.items()}
    for i in ids:
        k=freq(W["fc1.weight"][i,:p])
        W["fc1.weight"][i,:p]=proj(W["fc1.weight"][i,:p],k); W["fc1.weight"][i,p:]=proj(W["fc1.weight"][i,p:],k)
        W["fc2.weight"][:,i]=proj(W["fc2.weight"][:,i],k)
    return W
mem=[0,9,10,12,13,15,17]
alive=[int(i) for i in np.load(D+"scratch/alive.npy")]
single=[109,60,54,78,46,16,108,23,11,18,117]
C={"C1 mem7":mem,"C2 +54,60":mem+[54,60],"C3 +all singles":mem+single,"C4 +54,60,108,109":mem+[54,60,108,109]}
for name,ids in C.items():
    Wk=kill(W,ids); a,l=met(fwd(Wk)[1])
    rest=[i for i in alive if i not in ids]
    Wc=clean(Wk,rest); hc,Lc=fwd(Wc); ac,lc=met(Lc)
    off=Lc.mean(0); ao,lo=met(Lc-off)
    print(f"{name:22s} actual acc={a:.4f} loss={l:.3f} | clean acc={ac:.4f} loss={lc:.3f} | clean-offcancel acc={ao:.4f} loss={lo:.3f} offstd={off.std():.2f} maxb2={np.abs(Wk['fc2.bias']).max():.2f}")
