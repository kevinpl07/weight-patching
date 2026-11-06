import numpy as np
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64); W2=W["fc2.weight"].astype(np.float64); b2=W["fc2.bias"].astype(np.float64)
Q=np.load(S+"scratch/proj.npz"); P1,P2=Q["P1"],Q["P2"]
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
j=np.arange(p)
def ev(W1,b1,W2,name):
    H=np.maximum(X@W1.T+b1,0); L=H@W2.T; L=L-(H.mean(0)@W2.T)
    corr=L[np.arange(len(y)),y]; o=L.copy(); o[np.arange(len(y)),y]=-1e9
    m=corr-o.max(1)
    print(f"{name:50s} acc={(m>0).mean():.4f} medmargin={np.median(m):.2f} p5={np.percentile(m,5):.2f}")
def comp(v,k):
    F=np.fft.rfft(v); return np.abs(F[k]), np.angle(F[k]), F[0].real/len(v)
def cosv(k,amp,ph,dc=0):  # inverse of rfft single component
    return dc + 2*amp/p*np.cos(2*np.pi*k*j/p+ph)
def partner(n,k,da,db,do):
    Aa,pa,_=comp(W1[n,:p],k); Ab,pb,_=comp(W1[n,p:],k); Ao,po,_=comp(W2[:,n],k)
    row=np.concatenate([cosv(k,Aa,pa+da),cosv(k,Ab,pb+db)])
    col=cosv(k,Ao,po+do)
    return row,col
cands={
 "k41 partner of 52 (round2 style)":[(52,41,np.pi/2,-np.pi/2,0)],
 "k14 partner of 85 (round2 style)":[(85,14,np.pi/2,-np.pi/2,0)],
 "k41+k14":[(52,41,np.pi/2,-np.pi/2,0),(85,14,np.pi/2,-np.pi/2,0)],
 "k6+k32":[(29,6,np.pi/2,-np.pi/2,0),(24,32,np.pi/2,-np.pi/2,0)],
 "singletons k1+k20":[(53,1,np.pi/2,-np.pi/2,0),(67,20,np.pi/2,-np.pi/2,0)],
}
slots=[1,9]
ev(P1,b1,P2,"clean baseline")
for name,lst in cands.items():
    A1=P1.copy(); A2=P2.copy(); bb=b1.copy()
    for s,(n,k,da,db,do) in zip(slots,lst):
        r,c=partner(n,k,da,db,do); A1[s]=r; A2[:,s]=c; bb[s]=b1[n]
    ev(A1,bb,A2,name)
    bb2=bb.copy(); al=[i for i in range(128) if np.abs(A2[:,i]).sum()>1]; bb2[al]-=0.25
    ev(A1,bb2,A2,name+" +bias-0.25")
