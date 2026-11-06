import numpy as np
p=97
W=dict(np.load("weights.npz"))
W1=W["fc1.weight"];b1=W["fc1.bias"];W2=W["fc2.weight"];b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
def run(W1,b1,W2,b2):
    h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)  # 128 x N
    L=(W2@h).T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return -lp[np.arange(len(Y)),Y].mean(), (L.argmax(1)==Y).mean()
live=[i for i in range(128) if np.linalg.norm(W1[i])>1]
def conc(v):
    F=np.fft.rfft(v-v.mean()); e=np.abs(F)**2; e[0]=0; return (e/e.sum()).max(), e.argmax()
noisy=[i for i in live if conc(W1[i,:p])[0]<0.6 and conc(W2[:,i])[0]<0.6]
clean=[i for i in live if i not in noisy]
print("noisy",noisy, len(clean))
def proj(v,f,keepdc=True):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[f]=F[f]
    if keepdc: G[0]=F[0]
    return np.fft.irfft(G,n=p)
print("V0",run(W1,b1,W2,b2))
b1k=b1.copy(); b1k[noisy]=-4
print("V1 kill noisy",run(W1,b1k,W2,b2))
print("V2 +b2 const",run(W1,b1k,W2,np.full(p,b2.mean())))
print("V0 b2 const",run(W1,b1,W2,np.full(p,b2.mean())))
P1=W1.copy();P2=W2.copy()
for i in clean:
    f=conc(W1[i,:p])[1]
    P1[i,:p]=proj(W1[i,:p],f);P1[i,p:]=proj(W1[i,p:],f);P2[:,i]=proj(W2[:,i],f)
print("V3 projected",run(P1,b1k,P2,np.full(p,b2.mean())))
for s in [1.5,2,3]:
    print(" V3 scaled out",s,run(P1,b1k,P2*s,np.full(p,b2.mean())))
print("V3 projected but orig b2",run(P1,b1k,P2,b2))
# noisy neurons partially: ones with 0.4-0.6
for i in live:
    c=conc(W1[i,:p])[0]
    if 0.3<c<0.7: print("mid",i,c)
np.save("scratch/noisy.npy",np.array(noisy))
