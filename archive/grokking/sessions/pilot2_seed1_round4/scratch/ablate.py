import numpy as np
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
T=np.load(D+"train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
p=97
def fwd(W,A=A,B=B):
    W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
    pre=W1[:,A].T+W1[:,p+B].T+b1
    h=np.maximum(pre,0)
    return h, h@W2.T+b2
def ev(W,name):
    h,L=fwd(W)
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    print(f"{name:40s} acc={np.mean(L.argmax(1)==Y):.4f} loss={-lp[np.arange(len(Y)),Y].mean():.4f}")
ev(W,"orig")
mem=[0,9,10,12,13,15,17]
h,_=fwd(W)
def kill(W,ids,mean=False):
    W={k:v.copy() for k,v in W.items()}
    if mean:
        W["fc2.bias"]+=W["fc2.weight"][:,ids]@h[:,ids].mean(0)
    W["fc1.bias"][ids]=-10
    return W
ev(kill(W,mem),"zero-kill mem7")
ev(kill(W,mem,True),"mean-kill mem7")
for i in mem:
    ev(kill(W,[i],True),f"mean-kill {i}")
# clean projection
def proj(v,k):
    n=len(v); x=np.arange(n)
    c=np.cos(2*np.pi*k*x/n); s=np.sin(2*np.pi*k*x/n)
    return v.mean()+ c*(v@c)/(c@c)+ s*(v@s)/(s@s)
alive=[i for i in range(128) if np.abs(W["fc1.weight"][i]).sum()>0.1 and i not in mem]
def clean(W,ids,keepdc=True):
    W={k:v.copy() for k,v in W.items()}
    W1,W2=W["fc1.weight"],W["fc2.weight"]
    for i in ids:
        k=np.argmax(np.abs(np.fft.rfft(W1[i,:p]))[1:])+1
        W1[i,:p]=proj(W1[i,:p],k); W1[i,p:]=proj(W1[i,p:],k)
        W2[:,i]=proj(W2[:,i],k)
    return W
Wc=clean(kill(W,mem,True),alive)
ev(Wc,"clean all fourier + mem killed")
Wc2=kill(W,mem,True); 
# remove b2 too (clean proxy w/o per-class offsets): set b2 to its mean
h2,L2=fwd(Wc)
print("logit class-mean offsets std", L2.mean(0).std())
Wc3={k:v.copy() for k,v in Wc.items()}; Wc3["fc2.bias"]=Wc3["fc2.bias"]-L2.mean(0)
ev(Wc3,"clean + offsets cancelled")
np.save(D+"scratch/alive.npy",np.array(alive))
