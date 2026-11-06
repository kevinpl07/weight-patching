import numpy as np
D="<project>/runs/seed1/loop2/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
T=np.load(D+"train_pairs.npy"); a,b,y=T[:,0],T[:,1],T[:,2]
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T
off=L.mean(0)-L.mean()
print("no-b2 offset: std %.3f min %.3f max %.3f"%(off.std(),off.min(),off.max()))
offb=(L+b2).mean(0)-(L+b2).mean()
print("with b2 offset: std %.3f min %.3f max %.3f"%(offb.std(),offb.min(),offb.max()))
print("b2:",np.round(b2[:20],2))
print("label counts min/max",np.bincount(y,minlength=p).min(),np.bincount(y,minlength=p).max())
o=np.load(D+"optimizer.npz"); print(list(o.keys())[:12])
dead=np.where(np.abs(W2).sum(0)<0.1)[0]
for k in ['exp_avg/fc2.weight','exp_avg_sq/fc2.weight']:
    print(k, np.abs(o[k][:,dead]).max(), np.abs(o[k][:,np.setdiff1d(np.arange(128),dead)]).mean())
print('bias exp_avg_sq dead',np.abs(o['exp_avg_sq/fc1.bias'][dead]).max())
print(dead.tolist())
