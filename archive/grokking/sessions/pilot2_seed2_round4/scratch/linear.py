import numpy as np
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
I=np.load(D+"scratch/ideal.npz"); clean=I['clean']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
pre=W1[:,a].T+W1[:,p+b].T
print("noisy preact min (no bias):",pre[:,~clean].min())
for beta in [1.0,1.5,2.0,2.5]:
    bb=b1.copy(); bb[~clean]=beta
    h=np.maximum(pre+bb,0)
    c2=-(W2@h.mean(0)); c2-=c2.mean(); lg=h@W2.T+c2
    hn=h*(~clean); noise=(hn-hn.mean(0))@W2.T
    print(beta,"acc",(lg.argmax(1)==y).mean(),"noisy input-dep logit std",noise.std(),"c2 std",c2.std())
