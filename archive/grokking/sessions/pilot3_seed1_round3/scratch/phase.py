import numpy as np, sys
S="<project>/runs/seed1/loop2/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
A=W1[:,:p]; B=W1[:,p:]; O=W2.T
alive=np.where(np.linalg.norm(A,axis=1)>1)[0]
def ph(v,k):
    F=np.fft.rfft(v); return np.angle(F[k]), np.abs(F[k])
groups={}
for n in alive:
    tot=sum(np.abs(np.fft.rfft(M[n]-M[n].mean())[1:])**2 for M in (A,B,O)); k=np.argmax(tot)+1
    pa,_=ph(A[n],k); pb,_=ph(B[n],k); po,_=ph(O[n],k)
    groups.setdefault(k,[]).append((n,np.angle(np.exp(1j*(po-pa-pb))),np.angle(np.exp(1j*(pa-pb)))))
for k in sorted(groups):
    print(k, [(n,round(e,2),round(d,2)) for n,e,d in groups[k]])
# logits on train
d=np.load(S+"train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
h=np.maximum(A[:,a].T+B[:,b].T+b1,0); L=h@W2.T+b2
corr=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9
print("correct logit mean",corr.mean(),"margin mean",(corr-L2.max(1)).mean(),"min",(corr-L2.max(1)).min())
print("fc2.bias",np.round(b2,2))
cnt=np.bincount(y,minlength=p); print("label counts corr with bias", np.corrcoef(cnt,b2)[0,1])
L0=L-b2; m0=L0[np.arange(len(y)),y]; L0[np.arange(len(y)),y]=-1e9
print("acc no bias", ( (m0-L0.max(1))>0).mean(), "margin mean", (m0-L0.max(1)).mean())
