import numpy as np
D="<project>/runs/seed0/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); p=97
tr=np.load(D+"train_pairs.npy"); A_,B_,Y=tr[:,0],tr[:,1],tr[:,2]
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]
print("dead preact max:", end=" ")
H=W1[:,A_]+W1[:,p+B_]+b1[:,None]
live=[n for n in range(128) if np.linalg.norm(W1[n])>1]
dead=[n for n in range(128) if n not in live]
print(H[dead].max())
def kof(n): P=np.abs(np.fft.rfft(W1[n,:p]))**2; return np.argmax(P[1:])+1
h=np.maximum(H,0)
c=np.arange(p)
def decomp(L):
    # L: (S,p). signal g(d), d=(a+b-c)%p
    d=(A_[:,None]+B_[:,None]-c[None,:])%p
    L=L-L.mean(1,keepdims=True)
    g=np.bincount(d.ravel(),L.ravel(),p)/np.bincount(d.ravel(),minlength=p)
    S=g[d]; R=L-S
    # a,c part of residual
    fa=np.zeros((p,p)); cnt=np.bincount(A_,minlength=p)
    for a in range(p): fa[a]=R[A_==a].mean(0)
    Ra=fa[A_]; R2=R-Ra
    fb=np.zeros((p,p))
    for b in range(p): fb[b]=R2[B_==b].mean(0)
    Rb=fb[B_]; R3=R2-Rb
    e=lambda X:(X**2).sum()
    tot=e(L)
    return dict(signal=e(S)/tot, a_c=e(Ra)/tot, b_c=e(Rb)/tot, rest=e(R3)/tot, sig_peak=g[0]-np.sort(g)[-2])
L=h.T@W2.T
print("all", {k:round(float(v),3) for k,v in decomp(L).items()})
groups={}
for n in live: groups.setdefault(kof(n),[]).append(n)
for k,ns in sorted(groups.items()):
    Lk=h[ns].T@W2[:,ns].T
    print(k,ns,{kk:round(float(v),3) for kk,v in decomp(Lk).items()})
