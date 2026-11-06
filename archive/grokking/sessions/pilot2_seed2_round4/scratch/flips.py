import numpy as np
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
I=np.load(D+"scratch/ideal.npz"); clean=I['clean']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
A=W1[:,:p].copy();B=W1[:,p:].copy();O=W2.T.copy()
def ev(W1_,W2_,mask,label):
    h=np.maximum(W1_[:,a].T+W1_[:,p+b].T+b1,0)*mask
    c2=-(W2_@h.mean(0)); c2-=c2.mean(); lg=h@W2_.T+c2
    l=lg-lg.max(1,keepdims=True); lp=l-np.log(np.exp(l).sum(1,keepdims=True))
    c=lg[np.arange(len(y)),y]; l2=lg.copy(); l2[np.arange(len(y)),y]=-1e9; m=c-l2.max(1)
    print(f"{label:34s} acc {(lg.argmax(1)==y).mean():.4f} loss {-lp[np.arange(len(y)),y].mean():.3f} margin med {np.median(m):.2f} p5 {np.percentile(m,5):.2f}")
mask=clean.astype(float)
for th in [0.5,0.4,0.3,0.2]:
    tot=0; Xs=[]
    for X,C in [(A,I['CA']),(B,I['CB']),(O,I['CO'])]:
        mag=np.median(np.abs(X),1,keepdims=True); tgt=np.clip(2*C,-1,1)*mag
        sel=(np.sign(X)!=np.sign(C))&(np.abs(C)>th)&(np.abs(X)>0.2)&clean[:,None]
        X2=X.copy(); X2[sel]=tgt[sel]; tot+=sel.sum(); Xs.append(X2)
    ev(np.concatenate(Xs[:2],1),Xs[2].T,mask,f"fix flips |cos|>{th} n={tot}")
