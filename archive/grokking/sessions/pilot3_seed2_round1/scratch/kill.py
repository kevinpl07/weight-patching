import numpy as np
p=97
W=dict(np.load("weights.npz"))
w1=W["fc1.weight"]; b1=W["fc1.bias"]; w2=W["fc2.weight"]; b2=W["fc2.bias"]
print("fc2.bias", np.round(b2,2))
D=np.load("train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
A=w1[:,:p];B=w1[:,p:]
frac,k=np.load("scratch/frac.npy"); k=k.astype(int)
# freq clusters among neurons with frac>0.3
from collections import Counter
c=Counter(k[frac>0.3]); print(sorted(c.items(), key=lambda t:-t[1]))
c2=Counter(k[(frac>0.15)]); print(sorted(c2.items(), key=lambda t:-t[1]))
print("count frac<0.12",(frac<0.12).sum(),"<0.15",(frac<0.15).sum(),"<0.2",(frac<0.2).sum())
print("corr bias vs frac", np.corrcoef(b1,frac)[0,1])
