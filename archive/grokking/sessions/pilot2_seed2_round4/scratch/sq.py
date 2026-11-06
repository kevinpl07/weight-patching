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
clean=conc>0.6
x=np.arange(p)
def ideal(X,F):
    S=np.zeros_like(X); C=np.zeros_like(X)
    for n in range(128):
        c=np.cos(2*np.pi*f[n]*x/p+np.angle(F[n,f[n]]))
        mag=np.median(np.abs(X[n])); S[n]=np.sign(c)*mag; C[n]=c
    return S,C
SA,CA=ideal(A,FA);SB,CB=ideal(B,FB);SO,CO=ideal(O,FO)
for nm,X,S,C in [("A",A,SA,CA),("B",B,SB,CB),("O",O,SO,CO)]:
    dev=np.abs(X-S)[clean]; Cc=np.abs(C[clean])
    print(nm,"entries |dev|>0.5:",(dev>0.5).sum(),">1.0:",(dev>1.0).sum()," of those with |cos|>0.2:",((dev>1.0)&(Cc>0.2)).sum())
def ev(W1_,W2_,mask,label):
    h=np.maximum(W1_[:,a].T+W1_[:,p+b].T+b1,0)*mask
    c2=-(W2_@h.mean(0)); c2-=c2.mean(); lg=h@W2_.T+c2
    l=lg-lg.max(1,keepdims=True); lp=l-np.log(np.exp(l).sum(1,keepdims=True))
    c=lg[np.arange(len(y)),y]; l2=lg.copy(); l2[np.arange(len(y)),y]=-1e9; m=c-l2.max(1)
    print(f"{label:30s} acc {(lg.argmax(1)==y).mean():.4f} loss {-lp[np.arange(len(y)),y].mean():.3f} margin med {np.median(m):.2f} p5 {np.percentile(m,5):.2f}")
    return c2
mask=clean.astype(float)
ev(W1,W2,mask,"clean raw")
ev(np.concatenate([SA,SB],1),SO.T,mask,"clean ideal square")
# clipped cos ideal: clip(k*cos) to mag
for k in [2,3,5]:
    def cl(X,C): 
        mag=np.median(np.abs(X),1,keepdims=True); return np.clip(k*C,-1,1)*mag
    ev(np.concatenate([cl(A,CA),cl(B,CB)],1),cl(O,CO).T,mask,f"clean clipped-cos k={k}")
np.savez(D+"scratch/ideal.npz",SA=SA,SB=SB,SO=SO,CA=CA,CB=CB,CO=CO,clean=clean,f=f)
