import numpy as np
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
W1,W2=W["fc1.weight"],W["fc2.weight"]
p=97
for i in [1,66,110,31,109,54,0,9]:
    for name,v in [("a",W1[i,:p]),("b",W1[i,p:]),("o",W2[:,i])]:
        F=np.abs(np.fft.rfft(v))**2; F[1:]*=2
        k=np.argmax(F[1:])+1; order=np.argsort(F[1:])[::-1][:4]+1
        k3=min(3*k%p,p-3*k%p)
        print(i,name,"k",k,"top",order,"3k=",k3,"frac3k %.3f"%(F[k3]/F.sum()),"absmax %.2f"%np.abs(v).max(),"frac|v|>0.9 %.2f"%np.mean(np.abs(v)>0.9), "rms %.2f"%v.std())
