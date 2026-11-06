import numpy as np
exec(open("<project>/runs/seed0/loop/round2/agent/scratch/spur.py").read().split("print(\"four\"")[0])
Ltot=h@w2.T+b2; Ltot=Ltot-Ltot.mean(1,keepdims=True); T=(Ltot**2).sum()
def en(L,key):
    L=L-L.mean(1,keepdims=True); d=keys[key]
    g=np.bincount(d.ravel(),L.ravel(),p)/np.bincount(d.ravel(),minlength=p); return g
for k in sorted(set(kk[four])):
    idx=four[kk[four]==k]; L=h[:,idx]@w2[:,idx].T
    gm=en(L,"a+b-c"); gp=en(L,"a+b+c")
    Fm=np.fft.fft(gm); Fp=np.fft.fft(gp)
    print(f"k={k:2d} idx={list(idx)} alg_amp(k)={abs(Fm[k])*2/p:.2f} sum_amp(k)={abs(Fp[k])*2/p:.2f}  alg_margin={gm[0]-gm.mean():.2f}")
# partner check
w=w1/np.linalg.norm(w1,axis=1,keepdims=True); G=w@w.T
for i in four:
    j=np.argmin(G[i]); 
    if G[i,j]<-0.5: print("neg partner",i,j,round(G[i,j],2), "out corr", round(np.corrcoef(w2[:,i],w2[:,j])[0,1],2))
