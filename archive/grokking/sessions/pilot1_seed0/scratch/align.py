import numpy as np
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]
Wa=W1[:,:p]; Wb=W1[:,p:]
Fa=np.fft.fft(Wa,axis=1); Fb=np.fft.fft(Wb,axis=1); Fo=np.fft.fft(W2.T,axis=1)
# trilinear term: sum_a,b,c Wa[a]Wb[b]Wo[c] * [c==a+b]  = (1/p^2) sum_k Fa_k Fb_k conj(Fo_k) (up to convention)
T=np.array([[ (Fa[i,k]*Fb[i,k]*np.conj(Fo[i,k])).real for k in range(1,p)] for i in range(128)])
na=np.linalg.norm(Wa,axis=1);nb=np.linalg.norm(Wb,axis=1);no=np.linalg.norm(W2,axis=0)
score=T.sum(1)/(na*nb*no)/p   # normalized
# direct check: sum_{a,b} Wa[a]Wb[b]Wo[(a+b)%p]
aa,bb=np.meshgrid(np.arange(p),np.arange(p),indexing='ij')
direct=np.array([(np.outer(Wa[i],Wb[i])*W2[(aa+bb)%p,i]).sum() for i in range(128)])/(na*nb*no)
order=np.argsort(-direct)
print(np.round(np.sort(direct)[::-1],3))
print("order",order.tolist())
np.save(S+"scratch/align.npy",direct)
