# build edit: N dead neurons -> constant "class-offset" units: fc1.bias=beta, fc2 col = -offset/(N*beta)
import numpy as np, json, sys
D="<project>/runs/seed1/loop2/round4/agent/"
N=int(sys.argv[1]); beta=float(sys.argv[2]); out=sys.argv[3]
W=dict(np.load(D+"weights.npz")); p=97
T=np.load(D+"train_pairs.npy"); a,b,y=T[:,0],T[:,1],T[:,2]
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2
M=np.ones_like(L,bool); M[np.arange(len(y)),y]=False
off=np.array([L[M[:,c],c].mean() for c in range(p)]); off-=off.mean()
print("offset std %.3f range %.3f %.3f"%(off.std(),off.min(),off.max()))
dead=np.where(np.abs(W2).sum(0)<0.1)[0][:N].tolist()
col=(-off/(N*beta)).round(5)
ops=[{"tensor":"fc1.bias","rows":dead,"cols":None,"op":"set","value":[beta]*N},
     {"tensor":"fc2.weight","rows":None,"cols":dead,"op":"set","value":[[float(v)]*N for v in col]}]
json.dump(ops,open(D+"scratch/"+out,"w"))
# check
W1e=W1.copy();b1e=b1.copy();W2e=W2.copy(); b1e[dead]=beta; W2e[:,dead]=col[:,None]
h=np.maximum(W1e[:,a].T+W1e[:,p+b].T+b1e,0); L2=h@W2e.T+b2
print("acc",(L2.argmax(1)==y).mean(),"loss",np.mean(np.log(np.exp(L2-L2.max(1,keepdims=True)).sum(1))+L2.max(1)-L2[np.arange(len(y)),y]))
print("orig loss",np.mean(np.log(np.exp(L-L.max(1,keepdims=True)).sum(1))+L.max(1)-L[np.arange(len(y)),y]))
print(dead)
