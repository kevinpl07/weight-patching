import numpy as np
from collections import defaultdict
p=97
W=dict(np.load("weights.npz"))
W1=W["fc1.weight"];b1=W["fc1.bias"];W2=W["fc2.weight"];b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
noisy=list(np.load("scratch/noisy.npy"))
live=[i for i in range(128) if np.linalg.norm(W1[i])>1]
clean=[i for i in live if i not in noisy]
def fft(v): return np.fft.rfft(v)
def proj(v,f):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[f]=F[f]; G[0]=F[0]; return np.fft.irfft(G,n=p)
grp=defaultdict(list); info={}
for i in clean:
    Fa,Fb,Fo=fft(W1[i,:p]),fft(W1[i,p:]),fft(W2[:,i])
    e=np.abs(Fa)**2;e[0]=0;f=e.argmax()
    s=np.sort(e)[::-1]/e.sum()
    pa,pb,po=np.angle(Fa[f]),np.angle(Fb[f]),np.angle(Fo[f])
    # rfft X[f]=sum x_k e^{-i th k}; x_k ~ |X| cos(th k + angle) . out phase relation: ph_o ?= ph_a+ph_b
    rel=np.angle(np.exp(1j*(po-pa-pb)))
    grp[f].append(i)
    info[i]=(f,abs(Fa[f]),abs(Fb[f]),abs(Fo[f]),pa,pb,po,rel,s[1],b1[i])
for f in sorted(grp):
    print("freq",f)
    for i in grp[f]:
        x=info[i]; print("  n%3d |a|%.1f |b|%.1f |o|%.1f pa%+.2f pb%+.2f po%+.2f rel%+.2f 2nd%.3f bias%+.3f"%((i,)+x[1:]))
# per-group accuracy on train with projected weights
def logits(idx,P=True):
    L=np.zeros((len(Y),p))
    for i in idx:
        f=info[i][0]
        wa=proj(W1[i,:p],f) if P else W1[i,:p]; wb=proj(W1[i,p:],f) if P else W1[i,p:]; wo=proj(W2[:,i],f) if P else W2[:,i]
        h=np.maximum(wa[A]+wb[B]+b1[i],0)
        L+=h[:,None]*wo[None,:]
    return L
for f in sorted(grp):
    L=logits(grp[f]); 
    corr=L[np.arange(len(Y)),Y]; L2=L.copy(); L2[np.arange(len(Y)),Y]=-1e9
    print("f%2d n=%d acc %.3f mean margin %.2f"%(f,len(grp[f]),(L.argmax(1)==Y).mean(),(corr-L2.max(1)).mean()))
