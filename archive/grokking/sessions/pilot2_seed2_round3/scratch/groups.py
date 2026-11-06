import numpy as np
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
H=np.maximum(X@W1.T+b1,0)
L=H@W2.T+b2
print("acc",(L.argmax(1)==y).mean())
srt=np.sort(L,1); corr=L[np.arange(len(y)),y]
other=L.copy(); other[np.arange(len(y)),y]=-1e9
margin=corr-other.max(1)
print("margin pct",np.percentile(margin,[0,1,5,10,25,50]))
print("fc2.bias stats",b2.mean(),b2.std(), "abs dead weights round1:", np.abs(W1[1]).max(), np.abs(W2[:,1]).max(), np.abs(W1[6]).max())
def kof(v):
    F=np.fft.rfft(v-v.mean()); return np.argmax(np.abs(F[1:])**2)+1
alive=[n for n in range(128) if (H[:,n]>0).mean()>0.05]
groups={}
for n in alive: groups.setdefault(kof(W2[:,n]),[]).append(n)
c=np.arange(p)
tot_sig=0
res=[]
for k,ns in sorted(groups.items()):
    Lk=H[:,ns]@W2[:,ns].T  # N x p
    Lk=Lk-Lk.mean(1,keepdims=True)
    ideal=np.cos(2*np.pi*k*((a+b)[:,None]-c[None,:])/p)
    A=(Lk*ideal).sum()/(ideal**2).sum()
    resid=Lk-A*ideal
    # phases
    ph=[]
    for n in ns:
        Fa=np.fft.rfft(W1[n,:p])[k]; Fb=np.fft.rfft(W1[n,p:])[k]
        ph.append((np.angle(Fa)-np.angle(Fb)))
    res.append((k,ns,A,np.sqrt((resid**2).mean()),np.sqrt((Lk**2).mean())))
    print(f"k={k:2d} n={len(ns)} A={A:6.2f} noiseRMS={np.sqrt((resid**2).mean()):6.2f} tot={np.sqrt((Lk**2).mean()):6.2f}  phi_a-phi_b={np.round(np.array(ph)%(2*np.pi),2)} neurons={ns}")
# combined
