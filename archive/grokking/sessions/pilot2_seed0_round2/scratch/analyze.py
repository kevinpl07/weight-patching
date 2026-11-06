import numpy as np
S="<project>/runs/seed0/loop/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
Fa=np.fft.rfft(w1[:,:p],axis=1); Fb=np.fft.rfft(w1[:,p:],axis=1); Fo=np.fft.rfft(w2.T,axis=1)
Pa=np.abs(Fa)**2; Pb=np.abs(Fb)**2; Po=np.abs(Fo)**2
Pa[:,0]=0;Pb[:,0]=0;Po[:,0]=0
from collections import Counter
fr=[]
for i in range(128):
    ka=Pa[i].argmax(); kb=Pb[i].argmax(); ko=Po[i].argmax()
    fa=Pa[i,ka]/Pa[i].sum(); fb=Pb[i,kb]/Pb[i].sum(); fo=Po[i,ko]/Po[i].sum()
    na=np.linalg.norm(w1[i,:p]); nb=np.linalg.norm(w1[i,p:]); no=np.linalg.norm(w2[:,i])
    fr.append((i,ka,kb,ko,fa,fb,fo,na,nb,no,b1[i]))
    print(f"{i:3d} k=({ka:2d},{kb:2d},{ko:2d}) frac=({fa:.2f},{fb:.2f},{fo:.2f}) norm=({na:.2f},{nb:.2f},{no:.2f}) b={b1[i]:+.2f}")
c=Counter([f[3] for f in fr if f[6]>0.5]); print(c)
tot=Po.sum(0); print("output power by freq top:", sorted([(round(tot[k],1),k) for k in range(1,49)],reverse=True)[:12])
print("fc2 bias", b2.std())
