import numpy as np
p=97
W=dict(np.load("weights.npz"))
W1=W["fc1.weight"];b1=W["fc1.bias"];W2=W["fc2.weight"];b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]; N=len(Y)
noisy=list(np.load("scratch/noisy.npy"))
live=[i for i in range(128) if np.linalg.norm(W1[i])>1]
clean=[i for i in live if i not in noisy]
def H(W1,b1): return np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)
def ev(W1,b1,W2,b2):
    L=(W2@H(W1,b1)).T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(N),Y].mean()),4), round(float((L.argmax(1)==Y).mean()),4)
def proj(v,f):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[f]=F[f]; G[0]=F[0]; return np.fft.irfft(G,n=p)
P1=W1.copy();P2=W2.copy()
for i in clean:
    F=np.fft.rfft(W1[i,:p]);e=np.abs(F)**2;e[0]=0;f=e.argmax()
    P1[i,:p]=proj(W1[i,:p],f);P1[i,p:]=proj(W1[i,p:],f);P2[:,i]=proj(W2[:,i],f)
h0=H(W1,b1).mean(1)
for t in [-0.3,-0.4,-0.5,-0.6]:
    bf=b1.copy(); bf[clean]+=t
    h1=H(W1,bf).mean(1)
    b2c=b2+W2@(h0-h1)
    bp=bf.copy(); bp[noisy]=-4
    # clean projected with b2c relative: use projected out weights compensation
    print("t",t,"full",ev(W1,bf,W2,b2),"full+b2comp",ev(W1,bf,W2,b2c), "nchanged b2", int((np.abs(b2c-b2)>0).sum()))
    hp0=H(P1,np.where(np.isin(np.arange(128),noisy),-4,b1)).mean(1); hp1=H(P1,bp).mean(1)
    print("    clean proj const b2",ev(P1,bp,P2,np.full(p,b2.mean())), " clean proj, b2 = -P2@mean h", ev(P1,bp,P2,-P2@hp1))
print("----")
bk=b1.copy(); bk[noisy]=-4
hp=H(P1,bk).mean(1)
print("t=0 clean proj, DCcancel b2", ev(P1,bk,P2,-P2@hp), " with actual b2", ev(P1,bk,P2,b2))
hf=H(W1,b1).mean(1)
dc=-W2@hf
print("corr actual b2 vs -W2@meanh(full)", np.corrcoef(b2,dc)[0,1], "std",b2.std(),dc.std())
dcc=-W2[:,clean]@hf[clean]; dcn=-W2[:,noisy]@hf[noisy]
print("corr b2 vs clean DC", np.corrcoef(b2,dcc)[0,1], " vs noisy DC", np.corrcoef(b2,dcn)[0,1], "std clean dc",dcc.std(),"noisy dc",dcn.std())
print("full model with b2 replaced by DC cancel (full)", ev(W1,b1,W2,dc))
print("total class logit bias in full model (b2 + W2@meanh) std", (b2+W2@hf).std())
