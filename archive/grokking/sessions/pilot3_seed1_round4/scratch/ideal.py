import numpy as np
D="<project>/runs/seed1/loop2/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
T=np.load(D+"train_pairs.npy"); a,b,y=T[:,0],T[:,1],T[:,2]
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
alive=np.where(np.abs(W2).sum(0)>0.1)[0]
x=np.arange(p)
def fit(v):
    c=np.fft.fft(v); c[0]=0; pw=np.abs(c[1:49])
    k=np.argmax(pw)+1; return k, 2*np.abs(c[k])/p, np.angle(c[k]), v.mean()
def fwd(W1,b1,W2,b2):
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2
    pred=L.argmax(1); Ly=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9
    marg=Ly-L2.max(1)
    lse=np.log(np.exp(L-L.max(1,keepdims=True)).sum(1))+L.max(1)
    return (pred==y).mean(), np.mean(lse-Ly), np.median(marg), marg
print("orig",fwd(W1,b1,W2,b2)[:3])
info={}
for n in alive:
    info[n]=[fit(W1[n,:p]),fit(W1[n,p:]),fit(W2[:,n])]
for mode in ["sq","cos","clip"]:
  for keepb2 in [True,False]:
    V1=W1.copy();V2=W2.copy()
    for n in alive:
        (ka,ra,pa,ma),(kb,rb,pb,mb),(ko,ro,po,mo)=info[n]
        k=ka
        for (r,ph,m),sl in [((ra,pa,ma),slice(0,p)),((rb,pb,mb),slice(p,2*p))]:
            c=np.cos(2*np.pi*k*x/p+ph)
            V1[n,sl]= {"sq":0.88*np.sign(c),"cos":r*c,"clip":np.clip(1.3*c,-.88,.88)}[mode]
        c=np.cos(2*np.pi*k*x/p+po)
        V2[:,n]={"sq":0.88*np.sign(c),"cos":ro*c,"clip":np.clip(1.3*c,-.88,.88)}[mode]
    acc,loss,mm,marg=fwd(V1,b1,V2,b2 if keepb2 else 0*b2)
    print(mode,"b2" if keepb2 else "nob2",acc,loss,mm)
# check phase consistency: phase_out vs pa+pb, and ka==kb==ko
bad=[n for n in alive if not (info[n][0][0]==info[n][1][0]==info[n][2][0])]
print("freq mismatch",bad)

print("---- offset analysis")
def logits(W1,b1,W2,b2):
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); return h@W2.T+b2
def acc(L): return (L.argmax(1)==y).mean()
L0=logits(W1,b1,W2,b2); off0=L0.mean(0); print("actual: class-mean logit std",off0.std(), "acc after removing",acc(L0-off0))
L0nb=logits(W1,b1,W2,0*b2); print("actual no b2 classmean std",L0nb.mean(0).std(),"corr with b2",np.corrcoef(L0nb.mean(0),b2)[0,1])
for mode in ["clip","sq","cos"]:
    V1=W1.copy();V2=W2.copy()
    for n in alive:
        (ka,ra,pa,ma),(kb,rb,pb,mb),(ko,ro,po,mo)=info[n]; k=ka
        for (r,ph),sl in [((ra,pa),slice(0,p)),((rb,pb),slice(p,2*p))]:
            c=np.cos(2*np.pi*k*x/p+ph); V1[n,sl]={"sq":0.88*np.sign(c),"cos":r*c,"clip":np.clip(1.3*c,-.88,.88)}[mode]
        c=np.cos(2*np.pi*k*x/p+po); V2[:,n]={"sq":0.88*np.sign(c),"cos":ro*c,"clip":np.clip(1.3*c,-.88,.88)}[mode]
    L=logits(V1,b1,V2,0*b2); off=L.mean(0)
    print(mode,"nob2",acc(L),"offset std",off.std(),"acc offset removed",acc(L-off),"clip b2 to 1:",acc(L-np.clip(off,-1,1)))
    # margin stats
    Lr=L-off; Ly=Lr[np.arange(len(y)),y]; Lr2=Lr.copy(); Lr2[np.arange(len(y)),y]=-1e9
    print("  signal mean",Ly.mean()," runner-up mean",Lr2.max(1).mean(), "logit std",Lr.std())
