# candidate: N constant units + correct the largest fc2 sign-disagreements; cols cancel the post-flip offset
import numpy as np, json, sys
D="<project>/runs/seed1/loop2/round4/agent/"
N=int(sys.argv[1]); nflip=int(sys.argv[2]); which=sys.argv[3]; out=sys.argv[4]; frac=float(sys.argv[5]) if len(sys.argv)>5 else 1.0
W=dict(np.load(D+"weights.npz")); p=97; x=np.arange(p)
T=np.load(D+"train_pairs.npy"); a,b,y=T[:,0],T[:,1],T[:,2]
W1,b1,W2,b2=[W[k].copy() for k in ['fc1.weight','fc1.bias','fc2.weight','fc2.bias']]
alive=np.where(np.abs(W2).sum(0)>0.1)[0]; dead=np.where(np.abs(W2).sum(0)<0.1)[0][:N].tolist()
def fitcos(v):
    c=np.fft.fft(v); c[0]=0; k=np.argmax(np.abs(c[1:49]))+1; return np.cos(2*np.pi*k*x/p+np.angle(c[k]))
def run(W1,b1,W2,b2):
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2
    M=np.ones_like(L,bool); M[np.arange(len(y)),y]=False
    off=np.array([L[M[:,c],c].mean() for c in range(p)]); off-=off.mean()
    loss=np.mean(np.log(np.exp(L-L.max(1,keepdims=True)).sum(1))+L.max(1)-L[np.arange(len(y)),y])
    return off,(L.argmax(1)==y).mean(),loss
cands=[]
for n in alive:
    if 'o' in which:
        c=fitcos(W2[:,n]); t=np.clip(1.3*c,-.88,.88)
        for i in range(p):
            if np.sign(W2[i,n])!=np.sign(c[i]) and abs(c[i])>0.08: cands.append((abs(W2[i,n]-t[i]),'fc2.weight',i,n,t[i]))
    if 'i' in which:
        for sl in [0,p]:
            c=fitcos(W1[n,sl:sl+p]); t=np.clip(1.3*c,-.88,.88)
            for i in range(p):
                if np.sign(W1[n,sl+i])!=np.sign(c[i]) and abs(c[i])>0.08: cands.append((abs(W1[n,sl+i]-t[i]),'fc1.weight',n,sl+i,t[i]))
cands.sort(key=lambda r:-r[0]); print("total candidates",len(cands))
ops=[]
for _,tn,r,cc,v in cands[:nflip]:
    (W2 if tn=='fc2.weight' else W1)[r,cc]=v; ops.append({"tensor":tn,"rows":[int(r)],"cols":[int(cc)],"op":"set","value":[[float(v)]]})
off,acc,loss=run(W1,b1,W2,b2); print("after flips: offset std %.3f range %.2f %.2f acc %.4f loss %.4f"%(off.std(),off.min(),off.max(),acc,loss))
col=(-frac*off/N).round(5)
b1[dead]=1.0; W2[:,dead]=col[:,None]
off2,acc,loss=run(W1,b1,W2,b2); print("with const units: offset std %.3f acc %.4f loss %.4f maxcol %.3f"%(off2.std(),acc,loss,np.abs(col).max()))
ops=[{"tensor":"fc1.bias","rows":dead,"cols":None,"op":"set","value":[1.0]*N},
     {"tensor":"fc2.weight","rows":None,"cols":dead,"op":"set","value":[[float(v)]*N for v in col]}]+ops
json.dump(ops,open(D+"scratch/"+out,"w")); print("entries",N*98+nflip)
