import numpy as np
S="<project>/runs/seed0/loop/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
h=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0)
norm=np.linalg.norm(w1,axis=1)
Pa=np.abs(np.fft.rfft(w1[:,:p],axis=1))**2;Pa[:,0]=0
frac=Pa.max(1)/Pa.sum(1); kk=Pa.argmax(1)
live=norm>1; mem=np.where(live&(frac<0.65))[0]; four=np.where(live&(frac>=0.65))[0]
C=np.arange(p)
keys={"a+b-c":(A[:,None]+B[:,None]-C)%p,"a-c":(A[:,None]-C+0*B[:,None])%p,"b-c":(B[:,None]-C+0*A[:,None])%p,
      "a+c":(A[:,None]+C+0*B[:,None])%p,"b+c":(B[:,None]+C+0*A[:,None])%p,"a-b+c":(A[:,None]-B[:,None]+C)%p,"b-a+c":(B[:,None]-A[:,None]+C)%p,
      "a+b+c":(A[:,None]+B[:,None]+C)%p,"2a-c":(2*A[:,None]-C+0*B[:,None])%p,"c":(C+0*A[:,None])%p}
def decomp(L):
    L=L-L.mean(1,keepdims=True); tot=(L**2).sum(); out={}
    for n,d in keys.items():
        g=np.bincount(d.ravel(),L.ravel(),p)/np.bincount(d.ravel(),minlength=p)
        out[n]=round(float((g[d]**2).sum()/tot),3)
    return out
print("four",decomp(h[:,four]@w2[:,four].T))
print("all",decomp(h@w2.T+b2))
print("mem",decomp(h[:,mem]@w2[:,mem].T))
for k in [18,46,30,1]:
    idx=four[kk[four]==k]; print(k,idx,decomp(h[:,idx]@w2[:,idx].T))
