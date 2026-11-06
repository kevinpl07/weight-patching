import numpy as np
S="<project>/runs/seed1/loop2/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W['fc1.weight'].copy(); b1=W['fc1.bias']; W2=W['fc2.weight'].copy(); b2=W['fc2.bias']
d=np.load(S+"train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
x=np.arange(p)
def ev(W1,W2,b2,tag):
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    c=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9; m=c-L2.max(1)
    print(tag,"loss",-lp[np.arange(len(y)),y].mean(),"acc",(m>0).mean(),"margin mean",m.mean(),"p5",np.percentile(m,5))
ev(W1,W2,b2,"orig")
alive=np.where(np.linalg.norm(W1[:,:p],axis=1)>1)[0]
C1=W1.copy(); C2=W2.copy()
for n in alive:
    A=W1[n,:p]; B=W1[n,p:]; O=W2[:,n]
    tot=sum(np.abs(np.fft.rfft(v-v.mean())[1:])**2 for v in (A,B,O)); k=np.argmax(tot)+1
    for v,setter in ((A,lambda t:C1.__setitem__((n,slice(0,p)),t)),(B,lambda t:C1.__setitem__((n,slice(p,2*p)),t)),(O,lambda t:C2.__setitem__((slice(None),n),t))):
        F=np.fft.rfft(v); c=np.cos(2*np.pi*k*x/p+np.angle(F[k])); setter(np.clip(1.3*c,-.88,.88))
ev(C1,C2,b2,"clean")
h=np.maximum(C1[:,a].T+C1[:,p+b].T+b1,0); off=C2@h.mean(0)
ev(C1,C2,np.clip(-off,-1,1),"clean+bias(-off clipped)")
ev(C1,C2,-off,"clean+bias(-off)")
