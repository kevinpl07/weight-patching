import numpy as np, json, sys
S="<project>/runs/seed1/loop2/round3/agent/"
p=97; d=np.load(S+"train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
W0=dict(np.load(S+"weights.npz")); x=np.arange(p)
def ev(W,tag):
    W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2; cm=L.mean(0)
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    c=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9; m=c-L2.max(1)
    print(tag,"loss %.4f acc %.4f margin %.3f p5 %.3f classmean std %.3f"%(-lp[np.arange(len(y)),y].mean(),(m>0).mean(),m.mean(),np.percentile(m,5),cm.std()))
    return h
def flips(W, tensors, thr=0.08):
    W1=W['fc1.weight']; W2=W['fc2.weight']; cands=[]
    for n in np.where(np.linalg.norm(W1[:,:p],axis=1)>1)[0]:
        A=W1[n,:p]; B=W1[n,p:]; O=W2[:,n]
        tot=sum(np.abs(np.fft.rfft(v-v.mean())[1:])**2 for v in (A,B,O)); k=np.argmax(tot)+1
        for name,v,off in (("A",A,0),("B",B,p),("O",O,None)):
            F=np.fft.rfft(v); c=np.cos(2*np.pi*k*x/p+np.angle(F[k])); t=np.clip(1.3*c,-.88,.88)
            for i in range(p):
                if np.sign(v[i])!=np.sign(c[i]) and abs(c[i])>thr:
                    if off is None: cands.append((abs(t[i]-v[i]),"fc2.weight",i,int(n),float(t[i])))
                    else: cands.append((abs(t[i]-v[i]),"fc1.weight",int(n),off+i,float(t[i])))
    cands=[c for c in cands if c[1] in tensors]; cands.sort(key=lambda z:-z[0]); return cands
def build(nflip, tensors, nb, beta=1.0):
    W={k:v.copy() for k,v in W0.items()}; ops=[]
    for _,t,r,c,v in flips(W0,tensors)[:nflip]:
        W[t][r,c]=v; ops.append({"tensor":t,"rows":[r],"cols":[c],"op":"set","value":[[v]]})
    W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
    dead=np.where(np.linalg.norm(W1[:,:p],axis=1)<1)[0]
    for j in dead[:nb]:
        h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); cm=(h@W2.T+b2).mean(0); need=-(cm-cm.mean())
        o=np.clip(need/beta,-0.95,0.95); b1[j]=beta; W2[:,j]=o
        ops.append({"tensor":"fc1.bias","rows":[int(j)],"op":"set","value":[beta]})
        ops.append({"tensor":"fc2.weight","rows":None,"cols":[int(j)],"op":"set","value":[[float(v)] for v in o]})
        ops.append({"tensor":"fc2.weight","rows":None,"cols":[int(j)],"op":"reset_optimizer"})
        ops.append({"tensor":"fc1.bias","rows":[int(j)],"op":"reset_optimizer"})
    return W,ops
if __name__=="__main__":
    ev(W0,"orig")
    for nflip,tens,nb in ((0,(),1),(0,(),2),(749,("fc1.weight",),0),(651,("fc1.weight",),1),(553,("fc1.weight",),2),(553,("fc1.weight","fc2.weight"),2),(651,("fc1.weight","fc2.weight"),1)):
        W,ops=build(nflip,tens,nb); ev(W,f"flips={nflip} {tens} nb={nb}")
    if len(sys.argv)>1:
        nflip=int(sys.argv[1]); tens=tuple(sys.argv[2].split(",")) if sys.argv[2]!="-" else (); nb=int(sys.argv[3])
        W,ops=build(nflip,tens,nb); ev(W,"CHOSEN"); json.dump(ops,open(sys.argv[4],"w"))
