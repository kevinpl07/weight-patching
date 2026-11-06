import numpy as np, json, sys
S="<project>/runs/seed1/loop2/round3/agent/"
W=dict(np.load(S+"working.npz")); p=97
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
d=np.load(S+"train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
def ev(W1,b1,W2,b2,tag):
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    c=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9; m=c-L2.max(1)
    print(tag,"loss %.4f acc %.4f margin %.3f p5 %.3f classmean std %.3f"%(-lp[np.arange(len(y)),y].mean(),(m>0).mean(),m.mean(),np.percentile(m,5),(h@W2.T+b2).mean(0).std()))
    return h
h=ev(W1,b1,W2,b2,"working")
off=W2@h.mean(0); need=-off-b2
print("need std",need.std(),"max",np.abs(need).max())
dead=np.where(np.linalg.norm(W1[:,:p],axis=1)<1)[0]
for nb,beta in ((1,1.0),(2,1.0),(1,0.9)):
    B1=b1.copy(); B2=W2.copy(); rem=need.copy()
    for j in dead[:nb]:
        B1[j]=beta; o=np.clip(rem/beta,-0.95,0.95); B2[:,j]=o; rem=rem-o*beta
    ev(W1,B1,B2,b2,f"nb={nb} beta={beta}")
