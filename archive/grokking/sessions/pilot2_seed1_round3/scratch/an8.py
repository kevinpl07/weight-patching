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
H=np.maximum(Wa[:,a].T+Wb[:,b].T+b1,0)
EH=H.mean(0)
Bc=(H@W2.T).mean(0); 
print("Bc range",Bc.min()-Bc.mean(),Bc.max()-Bc.mean(), "Bc+b2 range", (Bc+b2-(Bc+b2).mean()).min(),(Bc+b2-(Bc+b2).mean()).max())
cW2=W2.copy()
for i in alive: cW2[:,i]=proj(W2[:,i],K[i])
Bcc=(H@cW2.T).mean(0); Bcc-=Bcc.mean()
print("clean-fc2 offset range",Bcc.min(),Bcc.max(),"std",Bcc.std())
R=W2-cW2
rows=[]
for i in alive:
    off=EH[i]*R[:,i]; off-=off.mean()
    rows.append((np.linalg.norm(R[:,i]),np.linalg.norm(off),EH[i],K[i],i, np.corrcoef(off,-Bcc)[0,1]))
rows.sort(reverse=True)
for r in rows[:20]: print("resnorm %.2f offnorm %.2f EH %.2f k=%d n=%d corr(off,-Bclean)=%.2f"%r)
# group Z_k
groups={}
for i in alive: groups.setdefault(K[i],[]).append(i)
for k,ns in sorted(groups.items()):
    Z=sum(EH[i]*np.fft.rfft(cW2[:,i])[k]*2/P for i in ns)
    S=sum(EH[i]*abs(np.fft.rfft(cW2[:,i])[k])*2/P for i in ns)
    print("k=%d n=%d |Z|=%.2f  sum|.|=%.2f"%(k,len(ns),abs(Z),S))
