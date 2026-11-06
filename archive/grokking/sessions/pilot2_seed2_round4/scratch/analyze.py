import numpy as np, collections
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
A=W1[:,:p];B=W1[:,p:];O=W2.T
def spec(X):
    F=np.fft.rfft(X,axis=1);P=np.abs(F)**2;P[:,0]=0;return F,P
FA,PA=spec(A);FB,PB=spec(B);FO,PO=spec(O)
tot=PA+PB+PO; f=tot.argmax(1)
conc=np.array([(PA[n,f[n]]/PA[n].sum()+PB[n,f[n]]/PB[n].sum()+PO[n,f[n]]/PO[n].sum())/3 for n in range(128)])
clean=conc>0.6
print("clean",clean.sum(), "noisy", (~clean).sum())
print("freqs clean:",sorted(collections.Counter(f[clean]).items()))
def fwd(W1,b1,W2,b2,mask=None):
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)
    if mask is not None: h=h*mask
    return h@W2.T+b2, h
def met(lg):
    lg=lg-lg.max(1,keepdims=True); lp=lg-np.log(np.exp(lg).sum(1,keepdims=True))
    return (lg.argmax(1)==y).mean(), -lp[np.arange(len(y)),y].mean()
lg,h=fwd(W1,b1,W2,b2); print("orig",met(lg))
lg,h=fwd(W1,b1,W2,b2,clean); print("noisy masked",met(lg))
# margin decomposition
def margin(lg):
    c=lg[np.arange(len(y)),y]; l2=lg.copy(); l2[np.arange(len(y)),y]=-1e9; return c-l2.max(1)
lgc,_=fwd(W1,b1,W2,b2,clean); lgn,_=fwd(W1,b1,W2,b2,~clean)
print("logit corr-class mean: clean",lgc[np.arange(len(y)),y].mean()-lgc.mean(), "noisy",lgn[np.arange(len(y)),y].mean()-lgn.mean())
print("margin clean only", np.percentile(margin(lgc),[5,25,50,75]))
# Fourier projected clean circuit
def proj(X,F,fs):
    out=np.zeros_like(X)
    for n in range(X.shape[0]):
        G=np.zeros_like(F[n]); G[fs[n]]=F[n,fs[n]]; G[0]=F[n,0]
        out[n]=np.fft.irfft(G,n=p)
    return out
Ap=proj(A,FA,f);Bp=proj(B,FB,f);Op=proj(O,FO,f)
W1p=np.concatenate([Ap,Bp],1)
lg,h=fwd(W1p,b1,Op.T,b2,clean); print("proj clean",met(lg))
for db in [0,-0.25,-0.5,-1.0,-1.5]:
    lg,h=fwd(W1p,b1+db,Op.T,0*b2,clean); print("proj clean bias",db,met(lg), "scaled x2",met(2*lg))
    lg,h=fwd(W1,b1+db,W2,0*b2,clean); print("   raw clean bias",db,met(lg))
