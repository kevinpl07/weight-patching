import numpy as np
D="<project>/runs/seed1/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
Wa,Wb=W1[:,:P],W1[:,P:]
def spec(v): return np.abs(np.fft.rfft(v-v.mean()))**2
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]; return np.fft.irfft(G,P)
alive=[i for i in range(128) if np.linalg.norm(Wa[i])>1]
K={i:np.argmax((spec(Wa[i])+spec(Wb[i])+spec(W2[:,i]))[1:])+1 for i in alive}
def ev(L):
    acc=(L.argmax(1)==y).mean(); Ls=L-L.max(1,keepdims=True)
    lp=Ls-np.log(np.exp(Ls).sum(1,keepdims=True)); return "acc %.4f loss %.4f"%(acc,-lp[np.arange(len(y)),y].mean())
H=np.maximum(Wa[:,a].T+Wb[:,b].T+b1,0)
cWa,cWb,cW2=Wa.copy(),Wb.copy(),W2.copy()
for i in alive: cWa[i]=proj(Wa[i],K[i]); cWb[i]=proj(Wb[i],K[i]); cW2[:,i]=proj(W2[:,i],K[i])
Hc=np.maximum(cWa[:,a].T+cWb[:,b].T+b1,0)
Lfull=H@W2.T; Lclean=Hc@cW2.T
Bf=Lfull.mean(0); Bf-=Bf.mean()
# analytic clean offset using full-model activations with clean fc2 cols
Bcc=(H@cW2.T).mean(0); Bcc-=Bcc.mean()
Bcl=Lclean.mean(0); Bcl-=Bcl.mean()
b2c=b2-b2.mean()
print("full, current:",ev(Lfull+b2),"| clean proxy:",ev(Lclean+b2))
for lam in [0,0.25,0.5,0.75,1.0]:
    tgt=-( (1-lam)*Bf + lam*Bcc )   # desired total of b2+bias
    add=tgt-b2c
    print("lam %.2f add range [%.2f,%.2f] | full: %s | clean proxy: %s"%(lam,add.min(),add.max(),ev(Lfull+b2+add),ev(Lclean+b2+add)))
