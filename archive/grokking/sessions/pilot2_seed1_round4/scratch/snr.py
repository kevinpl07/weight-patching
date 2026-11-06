import numpy as np
from collections import defaultdict
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
T=np.load(D+"train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
p=97; x=np.arange(p)
def proj(v,k):
    c=np.cos(2*np.pi*k*x/p); s=np.sin(2*np.pi*k*x/p)
    return v.mean()+c*(v@c)/(c@c)+s*(v@s)/(s@s)
def coef(v,k):
    F=np.fft.fft(v)[k]*2/p; return np.abs(F),np.angle(F)
alive=[int(i) for i in np.load(D+"scratch/alive.npy")]
grp=defaultdict(list)
for i in alive: grp[np.argmax(np.abs(np.fft.rfft(W1[i,:p]))[1:])+1].append(i)
c=np.arange(p)
tot=np.zeros((len(Y),p)); totoff=np.zeros(p)
rows=[]
for k,ids in sorted(grp.items()):
    G=np.zeros((len(Y),p))
    for i in ids:
        ua,ub,wo=proj(W1[i,:p],k),proj(W1[i,p:],k),proj(W2[:,i],k)
        h=np.maximum(ua[A]+ub[B]+b1[i],0)
        G+=np.outer(h,wo)
    ideal=np.cos(2*np.pi*k*((A+B)[:,None]-c[None,:])/p)
    S=(G*ideal).sum()/(ideal*ideal).sum()
    off=G.mean(0)
    J=G-S*ideal-off
    tot+=G; totoff+=off
    info=[(i,round(coef(W1[i,:p],k)[1],2),round(coef(W1[i,p:],k)[1],2),round(coef(W2[:,i],k)[1],2),round(np.maximum(proj(W1[i,:p],k)[A]+proj(W1[i,p:],k)[B]+b1[i],0).mean(),2)) for i in ids]
    print(f"k={k:2d} n={len(ids)} S={S:5.2f} junk_rms={J.std():.2f} off_rms={(off-off.mean()).std():.2f}  {info}")
print("total offset std", totoff.std())
