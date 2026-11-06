import numpy as np
W=dict(np.load('weights.npz'))
p=97
d=np.load('train_pairs.npy'); a,b,y=d[:,0],d[:,1],d[:,2]
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)
L=h@W2.T+b2
cnt=np.bincount(y,minlength=p)
print('corr b2 vs count',np.corrcoef(b2,cnt)[0,1], 'count range',cnt.min(),cnt.max())
corr=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9; marg=corr-L2.max(1)
print('correct logit mean',corr.mean(),'std',L.std(1).mean(),'margin pct',np.percentile(marg,[1,5,25,50,75]))
Lnb=h@W2.T; corr2=Lnb[np.arange(len(y)),y]; L3=Lnb.copy(); L3[np.arange(len(y)),y]=-1e9
print('acc no bias',(Lnb.argmax(1)==y).mean(), 'margin no-bias pct',np.percentile(corr2-L3.max(1),[1,5,25,50]))
