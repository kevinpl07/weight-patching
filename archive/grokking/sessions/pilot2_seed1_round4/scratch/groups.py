import numpy as np
from collections import defaultdict
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
T=np.load(D+"train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
p=97
alive=list(np.load(D+"scratch/alive.npy"))
grp=defaultdict(list)
for i in alive:
    F=np.abs(np.fft.rfft(W1[i,:p]))**2; F[1:]*=2
    k=np.argmax(F[1:])+1
    s=np.sort(F[1:])[::-1]
    grp[k].append(i)
    Fo=np.abs(np.fft.rfft(W2[:,i]))**2; Fo[1:]*=2
    print(i,k,"in top2/3 frac %.2f %.2f %.2f"%(s[0]/F.sum(),s[1]/F.sum(),s[2]/F.sum()), "out top", np.round(np.sort(Fo[1:])[::-1][:3]/Fo.sum(),2), "out-harm2 k2=%d frac %.3f"%(min(2*k%p,p-2*k%p),Fo[min(2*k%p,p-2*k%p)]/Fo.sum()))
print({k:v for k,v in sorted(grp.items())})
print("b2",np.round(b2,2))
