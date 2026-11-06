import numpy as np, json, sys
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
I=np.load(D+"scratch/ideal.npz"); clean=I['clean']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
pre=W1[:,a].T+W1[:,p+b].T
noisy=np.where(~clean)[0]
sd=pre[:,noisy].std(0); order=noisy[np.argsort(sd)]
print("noisy preact std range",sd.min(),sd.max())
def build(K,beta=2.0,flips=False):
    w1=W1.copy(); bb=b1.copy(); w2=W2.copy()
    dcn=order[:K]; kill=[n for n in noisy if n not in dcn]
    bb[kill]=-4.0
    if flips:
        A=w1[:,:p];B=w1[:,p:];O=w2.T.copy()
        for X,C in [(A,I['CA']),(B,I['CB']),(O,I['CO'])]:
            mag=np.median(np.abs(X),1,keepdims=True); tgt=np.clip(2*C,-1,1)*mag
            sel=(np.sign(X)!=np.sign(C))&(np.abs(C)>0.3)&(np.abs(X)>0.2)&clean[:,None]
            X[sel]=tgt[sel]
        w2=O.T.copy()
    h=np.maximum(w1[:,a].T+w1[:,p+b].T+bb,0)
    hc=h*clean
    c2=-(w2@hc.mean(0)); c2-=c2.mean()
    if K>0:
        bb[dcn]=beta
        h=np.maximum(w1[:,a].T+w1[:,p+b].T+bb,0)
        hbar=h[:,dcn].mean(0).sum()
        w2[:,dcn]=np.clip(c2/hbar,-1,1)[:,None]
    h=np.maximum(w1[:,a].T+w1[:,p+b].T+bb,0)
    nb=-(w2@h.mean(0)); nb-=nb.mean()
    return w1,bb,w2,nb,dcn,kill
def ev(w1,bb,w2,b2_,label):
    h=np.maximum(w1[:,a].T+w1[:,p+b].T+bb,0); lg=h@w2.T+b2_
    l=lg-lg.max(1,keepdims=True); lp=l-np.log(np.exp(l).sum(1,keepdims=True))
    print(f"{label:40s} acc {(lg.argmax(1)==y).mean():.4f} loss {-lp[np.arange(len(y)),y].mean():.3f}")
if __name__=="__main__":
    for K in [0,4,5,6]:
        for fl in [False,True]:
            w1,bb,w2,nb,dcn,kill=build(K,flips=fl)
            ev(w1,bb,w2,nb,f"K={K} flips={fl} |b2|max={np.abs(nb).max():.2f}")
            ev(w1,bb,w2,np.clip(nb,-1,1),f"   b2 clipped to 1")
            if K>0:
                bh=bb.copy(); bh[dcn]=1.0
                h=np.maximum(w1[:,a].T+w1[:,p+b].T+bh,0)
                ev(w1,bh,w2,np.clip(nb,-1,1),f"   beta->1, b2 clipped")
    print("dc neurons",order[:6])
