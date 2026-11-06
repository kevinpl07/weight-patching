import numpy as np
S="<project>/runs/seed1/loop2/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
d=np.load(S+"train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)
off=W2@h.mean(0)
print("corr(b2, -off)",np.corrcoef(b2,-off)[0,1], "off std",off.std(), "b2 std", b2.std())
L=h@W2.T
print("mean logit per class std (no bias)", L.mean(0).std(), "with bias", (L+b2).mean(0).std())
