import numpy as np, json, sys
S="<project>/runs/seed0/loop2/round2/agent/"
W0=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def ev(W):
    H=np.maximum(0,W['fc1.weight'][:,a].T+W['fc1.weight'][:,p+b].T+W['fc1.bias'])
    L=H@W['fc2.weight'].T+W['fc2.bias']; L=L-L.max(1,keepdims=True)
    lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(y)),y].mean()),4), round(float((L.argmax(1)==y).mean()),4)
def parts(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; return F[0].real/p, np.fft.irfft(G,n=p)
live=np.where(np.linalg.norm(W0['fc1.weight'][:,:p],axis=1)>1)[0]
dead=[int(i) for i in np.where(np.linalg.norm(W0['fc1.weight'][:,:p],axis=1)<1)[0]]
K={}
for n in live:
    P=np.abs(np.fft.rfft(W0['fc1.weight'][n,:p]))**2; P[0]=0; K[n]=int(P.argmax())
partners=[int(x) for x in sys.argv[1].split(",")]
full=sys.argv[2]=="full"
W={k:v.copy() for k,v in W0.items()}; ops=[]
for i,n in enumerate(partners):
    t=dead[i]; k=K[n]
    da,fa=parts(W0['fc1.weight'][n,:p],k); db,fb=parts(W0['fc1.weight'][n,p:],k)
    inp=np.concatenate([da-fa, db-fb])
    W['fc1.weight'][t]=inp; W['fc1.bias'][t]=W0['fc1.bias'][n]
    ops+= [{"tensor":"fc1.weight","rows":[t],"cols":None,"op":"set","value":[inp.tolist()]},
           {"tensor":"fc1.bias","rows":[t],"cols":None,"op":"set","value":float(W0['fc1.bias'][n])}]
    if full:
        do,fo=parts(W0['fc2.weight'][:,n],k); out=do+fo
        W['fc2.weight'][:,t]=out
        ops.append({"tensor":"fc2.weight","rows":None,"cols":[t],"op":"set","value":[[float(v)] for v in out]})
    print("twin",t,"of",n,"k",k)
for tn in ["fc1.weight","fc1.bias","fc2.weight"]:
    pass
ops.append({"tensor":"fc1.weight","rows":dead[:len(partners)],"cols":None,"op":"reset_optimizer"})
ops.append({"tensor":"fc1.bias","rows":dead[:len(partners)],"cols":None,"op":"reset_optimizer"})
if full: ops.append({"tensor":"fc2.weight","rows":None,"cols":dead[:len(partners)],"op":"reset_optimizer"})
print("base",ev(W0),"new",ev(W))
json.dump(ops,open(S+"scratch/twins.json","w"))
