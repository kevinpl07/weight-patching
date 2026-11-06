import numpy as np
D="<project>/runs/seed1/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
H=np.maximum(W1[:,a].T+W1[:,P+b].T+b1,0)
Lnb=H@W2.T
Bc=Lnb.mean(0); Bc-=Bc.mean()
print("std of per-class mean logit w/o b2: %.3f, with b2: %.3f"%(Bc.std(), (Bc+b2-b2.mean()).std()))
print("corr(Bc,b2)=%.3f"%np.corrcoef(Bc,b2)[0,1])
L=Lnb+b2
srt=np.sort(L,1); print("mean margin %.2f, mean correct logit-mean %.2f"%((L[np.arange(len(y)),y]-srt[:,-2]).mean(), (L[np.arange(len(y)),y]-L.mean(1)).mean()))
print("std of logits across classes (per pair) %.2f"%L.std(1).mean())
print("class counts std", np.bincount(y,minlength=P).std(), np.bincount(y,minlength=P).mean())
for c in range(20): print(c, "%.2f %.2f %d"%(Bc[c],b2[c],(y==c).sum()))
