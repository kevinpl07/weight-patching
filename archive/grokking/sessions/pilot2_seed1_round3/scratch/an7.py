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
    lp=Ls-np.log(np.exp(Ls).sum(1,keepdims=True)); return round(acc,4),round(float(-lp[np.arange(len(y)),y].mean()),4)
def run(Wa,Wb,W2):
    H=np.maximum(Wa[:,a].T+Wb[:,b].T+b1,0); return H@W2.T
cWa,cWb,cW2=Wa.copy(),Wb.copy(),W2.copy()
for i in alive: cWa[i]=proj(Wa[i],K[i]); cWb[i]=proj(Wb[i],K[i]); cW2[:,i]=proj(W2[:,i],K[i])
for name,(A,B,C) in {"full":(Wa,Wb,W2),"clean":(cWa,cWb,cW2),"cleanfc1":(cWa,cWb,W2)}.items():
    L=run(A,B,C); Bc=L.mean(0)-L.mean()
    print(name,"b2:",ev(L+b2),"no b2:",ev(L),"cancel:",ev(L-Bc), "Bc std %.2f resid w b2 %.2f"%(Bc.std(),(Bc+b2-b2.mean()).std()))
    # how does Bc arise: predicted from E[h]
