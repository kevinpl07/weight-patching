import numpy as np
S="<project>/runs/seed0/loop/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
def fwd(W,ret=False):
    w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
    h=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0); lo=h@w2.T+b2
    lo2=lo-lo.max(1,keepdims=True); lp=lo2-np.log(np.exp(lo2).sum(1,keepdims=True))
    r=(round(float(-lp[np.arange(len(Y)),Y].mean()),4), round(float((lo.argmax(1)==Y).mean()),4))
    return (r,h,lo) if ret else r
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
norm=np.linalg.norm(w1,axis=1)
Pa=np.abs(np.fft.rfft(w1[:,:p],axis=1))**2;Pa[:,0]=0
frac=Pa.max(1)/Pa.sum(1); kk=Pa.argmax(1)
live=norm>1; mem=np.where(live&(frac<0.65))[0]; four=np.where(live&(frac>=0.65))[0]
def proj(v,ks):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[0]=F[0]
    for k in ks: G[k]=F[k]
    return np.fft.irfft(G,n=p)
def h3(k): t=(3*k)%p; return min(t,p-t)
OUTK=lambda k:[k,h3(k)]
_,H,_=fwd(W,True); mact=H.mean(0)
def clean(W, drop=(), ablate=True):
    V={k:v.copy() for k,v in W.items()}
    for i in list(mem)+list(drop):
        if ablate: V["fc2.bias"]+=mact[i]*V["fc2.weight"][:,i]
        V["fc2.weight"][:,i]=0
    for i in four:
        k=kk[i]
        V["fc1.weight"][i,:p]=proj(w1[i,:p],[k,h3(k)]); V["fc1.weight"][i,p:]=proj(w1[i,p:],[k,h3(k)])
        V["fc2.weight"][:,i]=proj(w2[:,i],OUTK(k))
    return V
print("orig",fwd(W))
V={k:v.copy() for k,v in W.items()}
for i in mem: V["fc2.bias"]+=mact[i]*V["fc2.weight"][:,i]; V["fc2.weight"][:,i]=0
print("mean-ablate mem",fwd(V))
print("clean (k,3k in; k out), mem ablated",fwd(clean(W)))
V=clean(W); V["fc2.bias"][:]=V["fc2.bias"].mean(); print("clean + flat b2",fwd(V))
# per-group drop in clean model
base=fwd(clean(W))
for k in sorted(set(kk[four])):
    idx=four[kk[four]==k]; print("drop group",k,len(idx),fwd(clean(W,drop=idx)))
