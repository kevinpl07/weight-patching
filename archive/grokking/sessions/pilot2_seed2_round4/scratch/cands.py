import numpy as np
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
A=W1[:,:p];B=W1[:,p:];O=W2.T
def spec(X):
    F=np.fft.rfft(X,axis=1);P=np.abs(F)**2;P[:,0]=0;return F,P
FA,PA=spec(A);FB,PB=spec(B);FO,PO=spec(O)
f=(PA+PB+PO).argmax(1)
conc=np.array([(PA[n,f[n]]/PA[n].sum()+PB[n,f[n]]/PB[n].sum()+PO[n,f[n]]/PO[n].sum())/3 for n in range(128)])
def run(mask,db,recenter=True,label=""):
    bb=b1+db
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+bb,0)*mask
    if recenter: c2=-(W2@h.mean(0)); c2-=c2.mean()
    else: c2=b2
    lg=h@W2.T+c2
    l=lg-lg.max(1,keepdims=True); lp=l-np.log(np.exp(l).sum(1,keepdims=True))
    acc=(lg.argmax(1)==y).mean(); loss=-lp[np.arange(len(y)),y].mean()
    c=lg[np.arange(len(y)),y]; l2=lg.copy(); l2[np.arange(len(y)),y]=-1e9; m=c-l2.max(1)
    print(f"{label:28s} n={int(mask.sum())} acc {acc:.4f} loss {loss:.3f} margin med {np.median(m):.2f} p10 {np.percentile(m,10):.2f}")
for th in [0.6,0.45,0.3,0.2]:
    mask=(conc>th).astype(float)
    run(mask,0,False,f"th{th} no recenter")
    for db in [0,-0.25,-0.5,-0.75,-1.0,-1.25]:
        run(mask,db*(conc>th),True,f"th{th} db{db}")
