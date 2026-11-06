import numpy as np
S="<project>/runs/seed0/loop/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
def fwd(W):
    w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
    h=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0); lo=h@w2.T+b2
    lo=lo-lo.max(1,keepdims=True); lp=lo-np.log(np.exp(lo).sum(1,keepdims=True))
    return -lp[np.arange(len(Y)),Y].mean(), (lo.argmax(1)==Y).mean(), lo
w1=W["fc1.weight"];w2=W["fc2.weight"]
norm=np.linalg.norm(w1,axis=1)
Pa=np.abs(np.fft.rfft(w1[:,:p],axis=1))**2;Pa[:,0]=0
frac=Pa.max(1)/Pa.sum(1)
live=norm>1
mem=np.where(live&(frac<0.65))[0]; four=np.where(live&(frac>=0.65))[0]
print("mem",list(mem),"four",len(four))
print("orig",fwd(W)[:2])
V={k:v.copy() for k,v in W.items()}; V["fc2.weight"][:,mem]=0; print("kill mem",fwd(V)[:2])
# kill only fully non-Fourier (frac<0.2)
m2=np.where(live&(frac<0.2))[0]; print("m2",list(m2))
V={k:v.copy() for k,v in W.items()}; V["fc2.weight"][:,m2]=0; print("kill m2",fwd(V)[:2])
# ceiling: project Fourier neurons onto own freq (in, out)
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]*0; return np.fft.irfft(G,n=p)
V={k:v.copy() for k,v in W.items()}; V["fc2.weight"][:,mem]=0
for i in four:
    k=Pa[i].argmax()
    V["fc1.weight"][i,:p]=proj(w1[i,:p],k)+w1[i,:p].mean(); V["fc1.weight"][i,p:]=proj(w1[i,p:],k)+w1[i,p:].mean()
    V["fc2.weight"][:,i]=proj(w2[:,i],k)+w2[:,i].mean()
print("proj all",fwd(V)[:2])
print("---")
for mode in ["in","out"]:
    V={k:v.copy() for k,v in W.items()}
    for i in four:
        k=Pa[i].argmax()
        if mode=="in":
            V["fc1.weight"][i,:p]=proj(w1[i,:p],k)+w1[i,:p].mean(); V["fc1.weight"][i,p:]=proj(w1[i,p:],k)+w1[i,p:].mean()
        else: V["fc2.weight"][:,i]=proj(w2[:,i],k)+w2[:,i].mean()
    print("proj",mode,fwd(V)[:2])
# residual spectrum structure
Pb=np.abs(np.fft.rfft(w1[:,p:],axis=1))**2;Pb[:,0]=0
for i in four[:12]:
    o=np.argsort(Pa[i])[::-1][:5]; print(i,[(int(k),round(Pa[i,k]/Pa[i].sum(),3)) for k in o])
