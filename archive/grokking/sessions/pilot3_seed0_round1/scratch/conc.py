import numpy as np
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; b1=W["fc1.bias"]; w2=W["fc2.weight"]; b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
pre=w1[:,a].T+w1[:,p+b].T+b1; h=np.maximum(pre,0)
A=w1[:,:p]; B=w1[:,p:]
FA=np.abs(np.fft.rfft(A,axis=1))**2; FB=np.abs(np.fft.rfft(B,axis=1))**2; FO=np.abs(np.fft.rfft(w2,axis=0)).T**2
tot=FA+FB+FO
# power in single best freq (shared across a,b,out)
best=tot[:,1:].argmax(1)+1
c=np.array([tot[i,best[i]]/tot[i,1:].sum() for i in range(128)])
order=np.argsort(-c)
for i in order: print(i,best[i],round(c[i],3),"premax",round(pre[:,i].max(),2),"act",round((pre[:,i]>0).mean(),2))
def ev(mask):
    lg=h[:,mask]@w2[:,mask].T+b2
    m=lg.max(1,keepdims=True); ls=lg-m-np.log(np.exp(lg-m).sum(1,keepdims=True))
    return round((lg.argmax(1)==y).mean(),3), round(-ls[np.arange(len(y)),y].mean(),3)
for t in [0.6,0.5,0.4,0.3,0.2,0.15]:
    print(t,(c>t).sum(),ev(c>t))
np.save("scratch/conc.npy",c)
