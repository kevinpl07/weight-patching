import numpy as np, json, sys
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"].copy(); w2=W["fc2.weight"].copy(); b1=W["fc1.bias"]; b2=W["fc2.bias"]
TH=float(sys.argv[1]) if len(sys.argv)>1 else 0.3
BUDGET=int(sys.argv[2]) if len(sys.argv)>2 else 745
AMP=1.3; CLIP=0.88
x=np.arange(p)
A=w1[:,:p]; B=w1[:,p:]; O=w2.T
FA=np.fft.rfft(A,axis=1); FB=np.fft.rfft(B,axis=1); FO=np.fft.rfft(O,axis=1)
def share(F):
    P=np.abs(F[:,1:])**2; return P/P.sum(1,keepdims=True)
S=share(FA)+share(FB)+share(FO)
plans=[]
for i in range(128):
    k=S[i].argmax()+1
    pa,pb,po=np.angle(FA[i,k]),np.angle(FB[i,k]),np.angle(FO[i,k])
    if abs(np.angle(np.exp(1j*(po-pa-pb))))>0.35: continue
    ch=[]
    for part,w,ph,off in [("a",A[i],pa,0),("b",B[i],pb,p),("o",O[i],po,None)]:
        c=np.cos(2*np.pi*k*x/p+ph)
        idx=np.where((np.sign(w)!=np.sign(c))&(np.abs(c)>TH))[0]
        for j in idx: ch.append((part,int(j),float(np.clip(AMP*c[j],-CLIP,CLIP))))
    plans.append((len(ch),i,k,ch))
plans.sort(key=lambda t:t[0])
used=0; ops=[]; chosen=[]
new1=w1.copy(); new2=w2.copy()
for cost,i,k,ch in plans:
    if used+cost>BUDGET: continue
    used+=cost; chosen.append((i,k,cost))
    for part,j,v in ch:
        if part=="a": new1[i,j]=v; ops.append({"tensor":"fc1.weight","rows":[i],"cols":[j],"op":"set","value":[[v]]})
        elif part=="b": new1[i,p+j]=v; ops.append({"tensor":"fc1.weight","rows":[i],"cols":[p+j],"op":"set","value":[[v]]})
        else: new2[j,i]=v; ops.append({"tensor":"fc2.weight","rows":[j],"cols":[i],"op":"set","value":[[v]]})
print("used",used,"neurons",len(chosen)); print(chosen)
from collections import Counter
print("freq counts of chosen:",sorted(Counter(k for _,k,_ in chosen).items()))
json.dump(ops,open("scratch/edit_flip.json","w"))
d=np.load("train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
def ev(w1,w2):
    H=np.maximum(w1[:,a]+w1[:,p+b]+b1[:,None],0).T; L=H@w2.T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return (L.argmax(1)==y).mean(), -lp[np.arange(len(y)),y].mean()
print("orig",ev(w1,w2),"new",ev(new1,new2))
cs=[i for i,_,_ in chosen]
H=np.maximum(new1[:,a]+new1[:,p+b]+b1[:,None],0).T
print("chosen-only acc (new)",( (H[:,cs]@new2[:,cs].T).argmax(1)==y).mean())
H0=np.maximum(w1[:,a]+w1[:,p+b]+b1[:,None],0).T
print("chosen-only acc (orig)",( (H0[:,cs]@w2[:,cs].T).argmax(1)==y).mean())
