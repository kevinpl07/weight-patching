import numpy as np
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
I=np.load(D+"scratch/ideal.npz"); clean=I['clean']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)*clean
c2=-(W2@h.mean(0)); c2-=c2.mean()
print("c2 std",c2.std(),"max",np.abs(c2).max())
for s in [1,0.8,0.6,0.37,0.14,0]:
    lg=h@W2.T+s*c2
    print(s,(lg.argmax(1)==y).mean())
# also: current full network DC per class from noisy neurons
hn=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)*(~clean)
dn=W2@hn.mean(0); print("noisy DC corr with c2:",np.corrcoef(dn+b2,c2)[0,1], "noisy DC std",(dn+b2-np.mean(dn+b2)).std())
