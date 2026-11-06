# N constant units (bias 1, col cancels class offset) + undo fc2 sign-flips that act as extra class offset
# (rows where fc2.bias is saturated and the flip pushes that class's logit in the same direction as the bias)
import numpy as np, json, sys
D="<project>/runs/seed1/loop2/round4/agent/"
N=int(sys.argv[1]); maxflip=int(sys.argv[2]); out=sys.argv[3]
W=dict(np.load(D+"weights.npz")); p=97; x=np.arange(p)
T=np.load(D+"train_pairs.npy"); a,b,y=T[:,0],T[:,1],T[:,2]
W1,b1,W2,b2=[W[k].copy() for k in ['fc1.weight','fc1.bias','fc2.weight','fc2.bias']]
alive=np.where(np.abs(W2).sum(0)>0.1)[0]; dead=np.where(np.abs(W2).sum(0)<0.1)[0][:N].tolist()
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); mh=h.mean(0)
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
    c=fitcos(W2[:,n]); t=np.clip(1.3*c,-.88,.88)
    for i in range(p):
        if np.sign(W2[i,n])!=np.sign(c[i]) and abs(c[i])>0.08 and abs(b2[i])>0.9 and (W2[i,n]-t[i])*np.sign(b2[i])>0:
            cands.append((abs(W2[i,n]-t[i])*mh[n],i,n,t[i]))
cands.sort(key=lambda r:-r[0]); print("offset-type fc2 flips:",len(cands))
ops=[]
for _,i,n,v in cands[:maxflip]:
    W2[i,n]=v; ops.append({"tensor":"fc2.weight","rows":[int(i)],"cols":[int(n)],"op":"set","value":[[float(v)]]})
off,acc,loss=run(W1,b1,W2,b2); print("after flips: offset std %.3f range %.2f %.2f acc %.4f loss %.4f"%(off.std(),off.min(),off.max(),acc,loss))
col=(-off/N).round(5); b1[dead]=1.0; W2[:,dead]=col[:,None]
off2,acc,loss=run(W1,b1,W2,b2); print("with const units: acc %.4f loss %.4f maxcol %.3f"%(acc,loss,np.abs(col).max()))
ops=[{"tensor":"fc1.bias","rows":dead,"cols":None,"op":"set","value":[1.0]*N},
     {"tensor":"fc2.weight","rows":None,"cols":dead,"op":"set","value":[[float(v)]*N for v in col]}]+ops
json.dump(ops,open(D+"scratch/"+out,"w")); print("entries",N*98+len(ops)-2)
