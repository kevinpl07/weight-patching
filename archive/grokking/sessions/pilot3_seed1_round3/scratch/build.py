import numpy as np, json, sys
S="<project>/runs/seed1/loop2/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
OP=np.load(S+"scratch/optim0.npz")
NFLIP=int(sys.argv[1]); ZB=int(sys.argv[2]); out=sys.argv[3]
W1=W['fc1.weight']; W2=W['fc2.weight']; b2=W['fc2.bias']
U1=OP['next_update/fc1.weight']; U2=OP['next_update/fc2.weight']
x=np.arange(p)
alive=np.where(np.linalg.norm(W1[:,:p],axis=1)>1)[0]
cands=[]  # (dev, tensor, r, c, target, towards)
for n in alive:
    A=W1[n,:p]; B=W1[n,p:]; O=W2[:,n]
    tot=sum(np.abs(np.fft.rfft(v-v.mean())[1:])**2 for v in (A,B,O)); k=np.argmax(tot)+1
    for name,v,off in (("A",A,0),("B",B,p),("O",O,None)):
        F=np.fft.rfft(v); c=np.cos(2*np.pi*k*x/p+np.angle(F[k]))
        t=np.clip(1.3*c,-0.88,0.88)
        for i in range(p):
            if np.sign(v[i])!=np.sign(c[i]) and abs(c[i])>0.08:
                if off is None:
                    cands.append((abs(t[i]-v[i]),"fc2.weight",i,int(n),float(t[i]),np.sign(U2[i,n])==np.sign(t[i]-v[i])))
                else:
                    cands.append((abs(t[i]-v[i]),"fc1.weight",int(n),off+i,float(t[i]),np.sign(U1[n,off+i])==np.sign(t[i]-v[i])))
cands.sort(key=lambda z:-z[0])
print("n cands",len(cands),"frac optimizer already moving toward target",np.mean([c[5] for c in cands]))
sel=cands[:NFLIP]
print("selected",len(sel),"min dev",sel[-1][0] if sel else None)
ops=[{"tensor":t,"rows":[r],"cols":[c],"op":"set","value":[[v]]} for _,t,r,c,v,_ in sel]
if ZB:
    ops.append({"tensor":"fc2.bias","rows":None,"op":"set","value":0.0})
    ops.append({"tensor":"fc2.bias","rows":None,"op":"reset_optimizer"})
json.dump(ops,open(out,"w"))
