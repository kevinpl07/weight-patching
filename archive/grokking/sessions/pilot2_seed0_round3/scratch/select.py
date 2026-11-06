import numpy as np
exec(open('proxy3.py').read().split('if __name__')[0])
h=hidden(W)  # 128 x S
act=(h>0)
W2=W["fc2.weight"]
R1=W["fc1.weight"]-P["fc1.weight"]; R2=W2-P["fc2.weight"]
# importance weights
w1=np.zeros_like(R1)
for n in live:
    cn=(W2[:,n]**2).sum()
    w1[n,:p]=np.bincount(A_[act[n]],minlength=p)*cn
    w1[n,p:]=np.bincount(B_[act[n]],minlength=p)*cn
w2=np.zeros_like(R2)
for n in live: w2[:,n]=(h[n]**2).sum()
E1=R1**2*w1; E2=R2**2*w2
print("total weighted resid energy fc1 %.0f fc2 %.0f"%(E1.sum(),E2.sum()))
allE=np.concatenate([E1.ravel(),E2.ravel()]); order=np.argsort(allE)[::-1]
for K in [300,500,650,749]:
    sel=order[:K]; n1=(sel<E1.size).sum()
    print(K,"frac captured %.3f"%(allE[sel].sum()/allE.sum()),"fc1 entries",n1,"fc2",K-n1)
o1=np.sort(E1.ravel())[::-1]; o2=np.sort(E2.ravel())[::-1]
for K in [650,749]: print(K,"fc1-only frac of fc1 %.3f"%(o1[:K].sum()/E1.sum()),"fc2-only frac of fc2 %.3f"%(o2[:K].sum()/E2.sum()))
np.savez("resid.npz",R1=R1,R2=R2,E1=E1,E2=E2)
