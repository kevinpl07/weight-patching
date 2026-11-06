import numpy as np
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
T=np.load(D+"train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
p=97; N=len(Y); r=np.arange(N)
h=np.maximum(W1[:,A].T+W1[:,p+B].T+b1,0)
mem=[0,9,10,12,13,15,17]
alive=[int(i) for i in np.load(D+"scratch/alive.npy")]
Lf=h[:,alive]@W2[:,alive].T; Lm=h[:,mem]@W2[:,mem].T
def met(L):
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(np.mean(L.argmax(1)==Y),4), round(-lp[r,Y].mean(),3)
def margin(L):
    c=L[r,Y].copy(); L2=L.copy(); L2[r,Y]=-1e9; return c-L2.max(1)
print("fourier offset std",Lf.mean(0).std(),"mem offset std",Lm.mean(0).std(),"b2 std",b2.std())
print("corr(fourier off, mem off+b2)",np.corrcoef(Lf.mean(0),Lm.mean(0)+b2)[0,1])
print("total offset std", (Lf.mean(0)+Lm.mean(0)+b2).std())
Lmv=Lm-Lm.mean(0)
print("mem varying part rms",Lmv.std(), "fourier varying rms",(Lf-Lf.mean(0)).std())
# correct-class: how much of the margin does each provide
for name,L in [("full",Lf+Lm+b2),("fourier+offsets(all)",Lf-Lf.mean(0)),("fourier+mem_mean+b2",Lf+Lm.mean(0)+b2),("fourier+b2",Lf+b2),("fourier + clip(b2+memmean)",Lf+np.clip(b2+Lm.mean(0),-1,1))]:
    m=margin(L); print(f"{name:30s}",met(L),"margin median %.2f  p10 %.2f"%(np.median(m),np.percentile(m,10)))
print("mem varying part at correct class mean %.2f; at others mean %.2f"%(Lmv[r,Y].mean(), (Lmv.sum(1)-Lmv[r,Y]).mean()/96))
