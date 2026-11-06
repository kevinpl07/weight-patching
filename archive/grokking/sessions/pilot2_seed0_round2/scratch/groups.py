import numpy as np
S="<project>/runs/seed0/loop/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
h=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0)  # N x 128
norm=np.linalg.norm(w1,axis=1)
Pa=np.abs(np.fft.rfft(w1[:,:p],axis=1))**2;Pa[:,0]=0
frac=Pa.max(1)/Pa.sum(1); kk=Pa.argmax(1)
live=norm>1; mem=np.where(live&(frac<0.65))[0]; four=np.where(live&(frac>=0.65))[0]
C=np.arange(p)
d=(A[:,None]+B[:,None]-C[None,:])%p
def stats(idx,name):
    L=h[:,idx]@w2[:,idx].T
    L=L-L.mean(1,keepdims=True)
    # algorithmic part: average over d
    g=np.array([L[d==j].mean() for j in range(p)])
    Lg=g[d]
    ve=1-((L-Lg)**2).sum()/(L**2).sum()
    marg=(L[np.arange(len(Y)),Y]-L.mean(1)).mean()
    gm=g[0]-g.mean()
    return ve,marg,gm,np.sqrt((L**2).mean())
print("all",stats(np.arange(128),"all"))
print("four",stats(four,"four"))
print("mem",stats(mem,"mem"))
for k in sorted(set(kk[four])):
    idx=four[kk[four]==k]; ve,m,gm,rms=stats(idx,k)
    print(f"k={k:2d} n={len(idx)} varexpl={ve:.2f} margin={m:6.2f} alg_margin={gm:6.2f} rms={rms:.2f}")
for i in mem:
    ve,m,gm,rms=stats(np.array([i]),i); print(f"mem {i} varexpl={ve:.2f} margin={m:6.2f} alg={gm:6.2f} rms={rms:.2f} active={np.mean(h[:,i]>0):.2f}")
cnt=np.bincount(Y,minlength=p); print("corr b2 vs label count",np.corrcoef(b2,cnt)[0,1], "b2 mean",b2.mean())
