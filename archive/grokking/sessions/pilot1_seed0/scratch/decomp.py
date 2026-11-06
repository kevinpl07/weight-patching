import numpy as np
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
Wa=W1[:,:p]; Wb=W1[:,p:]
al=np.load(S+"scratch/align.npy")
H=np.maximum(Wa[:,A].T+Wb[:,B].T+b1,0)
F=al>3; M=~F
n=np.arange(len(Y))
for name,mask in [("F",F),("M",M)]:
    lg=H[:,mask]@W2[:,mask].T
    cor=lg[n,Y]; lg2=lg.copy(); lg2[n,Y]=-1e9
    print(name,mask.sum(),"correct logit mean %.2f std %.2f | mean other %.2f | max other mean %.2f | logit std over classes %.2f"%(cor.mean(),cor.std(),np.delete(lg,0,1).mean(),lg2.max(1).mean(),lg.std(1).mean()))
lg=H@W2.T+b2
cor=lg[n,Y]; lg2=lg.copy(); lg2[n,Y]=-1e9
print("all: correct %.2f, max other %.2f, margin min %.2f"%(cor.mean(),lg2.max(1).mean(),(cor-lg2.max(1)).min()))
print("b2 std",b2.std())
# label frequency vs b2
cnt=np.bincount(Y,minlength=p); print("corr(b2, count)",np.corrcoef(b2,cnt)[0,1])
