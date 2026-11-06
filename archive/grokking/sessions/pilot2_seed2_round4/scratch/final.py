import numpy as np, json
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
I=np.load(D+"scratch/ideal.npz"); clean=I['clean']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
K=6; BETA=2.0
pre=W1[:,a].T+W1[:,p+b].T
noisy=np.where(~clean)[0]
order=noisy[np.argsort(pre[:,noisy].std(0))]
dcn=[int(n) for n in order[:K]]; kill=[int(n) for n in noisy if n not in dcn]
ops=[]
# 1) kill memorizing neurons
ops.append({"tensor":"fc1.bias","rows":kill,"cols":None,"op":"set","value":-4.0})
# 2) fix far-from-boundary sign flips in clean neurons to clipped-cos values
w1=W1.copy(); w2=W2.copy(); nflip=0
for name,X,C in [("A",w1[:,:p],I['CA']),("B",w1[:,p:],I['CB']),("O",w2.T,I['CO'])]:
    mag=np.median(np.abs(X),1,keepdims=True); tgt=np.clip(2*C,-1,1)*mag
    sel=(np.sign(X)!=np.sign(C))&(np.abs(C)>0.3)&(np.abs(X)>0.2)&clean[:,None]
    for n,j in zip(*np.where(sel)):
        v=float(tgt[n,j]); nflip+=1
        if name=="A": ops.append({"tensor":"fc1.weight","rows":[int(n)],"cols":[int(j)],"op":"set","value":[[v]]})
        if name=="B": ops.append({"tensor":"fc1.weight","rows":[int(n)],"cols":[int(j)+p],"op":"set","value":[[v]]})
        if name=="O": ops.append({"tensor":"fc2.weight","rows":[int(j)],"cols":[int(n)],"op":"set","value":[[v]]})
        X[n,j]=v
# 3) DC provider neurons: always-on (bias 2 > max |preact|), output row carries per-class offset
hc=np.maximum(w1[:,a].T+w1[:,p+b].T+b1,0)*clean
need=-(w2@hc.mean(0)) - b2      # offset needed on top of existing fc2.bias
need-=need.mean()
bb=b1.copy(); bb[kill]=-4.0; bb[dcn]=BETA
h=np.maximum(w1[:,a].T+w1[:,p+b].T+bb,0)
hbar=h[:,dcn].mean(0).sum()
Ocol=np.clip(need/hbar,-1,1)
print("dc neurons",dcn,"hbar",hbar,"Ocol max",np.abs(Ocol).max(),"flips",nflip)
ops.append({"tensor":"fc1.bias","rows":dcn,"cols":None,"op":"set","value":BETA})
for n in dcn:
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[n],"op":"set","value":[[float(v)] for v in Ocol]})
    w2[:,n]=Ocol
lg=h@w2.T+b2
l=lg-lg.max(1,keepdims=True); lp=l-np.log(np.exp(l).sum(1,keepdims=True))
print("pred acc",(lg.argmax(1)==y).mean(),"loss",-lp[np.arange(len(y)),y].mean())
print("entries",len(kill)+nflip+K+K*p)
json.dump(ops,open(D+"scratch/edit.json","w"))
