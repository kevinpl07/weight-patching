import numpy as np, sys
S="<project>/runs/seed0/loop2/round2/agent/"
W=dict(np.load(S+(sys.argv[1] if len(sys.argv)>1 else "weights.npz"))); p=97
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
H=np.maximum(0,W1[:,a].T+W1[:,p+b].T+b1)
L=H@W2.T+b2
off=H.mean(0)@W2.T
print("offset std",off.std(),"b2 std",b2.std(),"(off+b2) std",(off+b2).std(), "corr",np.corrcoef(off,b2)[0,1])
corr=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9; mx=L2.max(1)
m=corr-mx
print("margin pct",np.percentile(m,[5,25,50,75]),"acc",(m>0).mean(), "logit std", L.std(), "correct logit mean",corr.mean())
# how bad would predictions be with offset only
cnt=np.bincount(y,minlength=p); print("label count min/max",cnt.min(),cnt.max())
# label-dependent error: per-class accuracy vs (off+b2)
acc_c=np.array([(m[y==c]>0).mean() for c in range(p)])
print("corr(acc_c, off+b2)",np.corrcoef(acc_c,off+b2)[0,1])
pred=L.argmax(1); pc=np.bincount(pred,minlength=p); print("pred count min/max",pc.min(),pc.max(), "corr(predcount-labelcount, off+b2)",np.corrcoef(pc-cnt,off+b2)[0,1])
