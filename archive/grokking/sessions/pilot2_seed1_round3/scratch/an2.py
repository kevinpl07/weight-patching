import numpy as np
D="<project>/runs/seed1/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
Wa,Wb=W1[:,:P],W1[:,P:]
x=np.arange(P)
def proj(v,k,dc=True):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]
    if dc: G[0]=F[0]
    return np.fft.irfft(G,P)
def spec(v): return np.abs(np.fft.rfft(v-v.mean()))**2
alive=[i for i in range(128) if np.linalg.norm(Wa[i])>1]
K={}
for i in alive:
    t=spec(Wa[i])+spec(Wb[i])+spec(W2[:,i]); K[i]=np.argmax(t[1:])+1
def fwd(Wa,Wb,b1,W2,b2):
    H=np.maximum(Wa[:,a].T+Wb[:,b].T+b1,0); L=H@W2.T+b2
    acc=(L.argmax(1)==y).mean()
    Ls=L-L.max(1,keepdims=True); lp=Ls-np.log(np.exp(Ls).sum(1,keepdims=True))
    return acc,-lp[np.arange(len(y)),y].mean()
print("full",fwd(Wa,Wb,b1,W2,b2))
cWa,cWb,cW2=Wa.copy(),Wb.copy(),W2.copy()
for i in alive:
    cWa[i]=proj(Wa[i],K[i]); cWb[i]=proj(Wb[i],K[i]); cW2[:,i]=proj(W2[:,i],K[i])
print("clean fc1, full fc2",fwd(cWa,cWb,b1,W2,b2))
print("full fc1, clean fc2",fwd(Wa,Wb,b1,cW2,b2))
print("clean both",fwd(cWa,cWb,b1,cW2,b2))
# fc2 with harmonics allowed: project fc2 onto k,2k,3k
def projm(v,ks):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[0]=F[0]
    for k in ks:
        kk=k%P; kk=min(kk,P-kk); G[kk]=F[kk]
    return np.fft.irfft(G,P)
hW2=W2.copy()
for i in alive: hW2[:,i]=projm(W2[:,i],[K[i],2*K[i]])
print("clean fc1, fc2 k+2k",fwd(cWa,cWb,b1,hW2,b2))
# where is non-dominant power: harmonic share
for i in alive[:10]:
    for v,nm in [(Wa[i],'a'),(W2[:,i],'o')]:
        s=spec(v); k=K[i]; k2=min((2*k)%P,P-(2*k)%P); k3=min((3*k)%P,P-(3*k)%P)
        print(i,nm,k,"dom %.2f 2k %.3f 3k %.3f"%(s[k]/s.sum(),s[k2]/s.sum(),s[k3]/s.sum()))
