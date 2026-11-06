import numpy as np, json, sys
S="<project>/runs/seed0/loop2/round2/agent/"
W0=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def ev(W):
    H=np.maximum(0,W['fc1.weight'][:,a].T+W['fc1.weight'][:,p+b].T+W['fc1.bias'])
    L=H@W['fc2.weight'].T+W['fc2.bias']; L=L-L.max(1,keepdims=True)
    lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(y)),y].mean()),4), round(float((L.argmax(1)==y).mean()),4)
def parts(v,k,amp=None):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]
    if amp is not None: G[k]=G[k]/np.abs(G[k])*amp*p/2
    G[0]=F[0]; return np.fft.irfft(G,n=p)
live=[int(n) for n in np.where(np.linalg.norm(W0['fc1.weight'][:,:p],axis=1)>1)[0]]
K={}
for n in live:
    P=np.abs(np.fft.rfft(W0['fc1.weight'][n,:p]))**2; P[0]=0; K[n]=int(P.argmax())
C1=W0['fc1.weight'].copy(); C2=W0['fc2.weight'].copy()
for n in live:
    k=K[n]; C1[n,:p]=parts(W0['fc1.weight'][n,:p],k); C1[n,p:]=parts(W0['fc1.weight'][n,p:],k); C2[:,n]=parts(W0['fc2.weight'][:,n],k)
E1=np.zeros_like(C1); E1[live]=(W0['fc1.weight']-C1)[live]; E2=np.zeros_like(C2); E2[:,live]=(W0['fc2.weight']-C2)[:,live]
totnoise=(E1**2).sum()+(E2**2).sum()
mode=sys.argv[1]
W={k:v.copy() for k,v in W0.items()}; ops=[]
budget=749
if mode=="Y":
    for n in (82,97):
        k=K[n]
        fa=parts(W0['fc1.weight'][n,:p],k,1.0); fb=parts(W0['fc1.weight'][n,p:],k,1.0); fo=parts(W0['fc2.weight'][:,n],k,1.0)
        W['fc1.weight'][n]=np.concatenate([fa,fb]); W['fc2.weight'][:,n]=fo
        E1[n]=0; E2[:,n]=0
    budget-=582
# top entries by |noise|
flat=np.concatenate([np.abs(E1).ravel(),np.abs(E2).ravel()])
idx=np.argsort(flat)[::-1][:budget]
n1=E1.size
for i in idx:
    if i<n1:
        r,c=divmod(i,E1.shape[1]); W['fc1.weight'][r,c]=C1[r,c]
    else:
        r,c=divmod(i-n1,E2.shape[1]); W['fc2.weight'][r,c]=C2[r,c]
removed=sum(((W0[k]-W[k])**2).sum() for k in W0)
print(mode,"train",ev(W),"changed",sum(int((W0[k]!=W[k]).sum()) for k in W0),"noise frac removed~",round(float(removed/totnoise),3))
# build ops
for t in ("fc1.weight","fc2.weight"):
    rr,cc=np.where(W0[t]!=W[t])
    for r,c in zip(rr,cc):
        ops.append({"tensor":t,"rows":[int(r)],"cols":[int(c)],"op":"set","value":[[float(W[t][r,c])]]})
json.dump(ops,open(S+f"scratch/clean_{mode}.json","w"))
