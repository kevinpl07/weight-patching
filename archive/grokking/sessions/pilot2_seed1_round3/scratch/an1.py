import numpy as np
D="<project>/runs/seed1/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
Wa,Wb=W1[:,:P],W1[:,P:]
def spec(v): 
    F=np.fft.rfft(v-v.mean()); return np.abs(F)**2
H=np.maximum(Wa[:,a].T+Wb[:,b].T+b1,0)
act=(H>0).mean(0)
rows=[]
for i in range(128):
    sa,sb,so=spec(Wa[i]),spec(Wb[i]),spec(W2[:,i])
    tot=sa+sb+so
    k=np.argmax(tot[1:])+1
    fa,fb,fo=sa[k]/sa.sum(),sb[k]/sb.sum(),so[k]/so.sum()
    rows.append((i,k,fa,fb,fo,np.linalg.norm(Wa[i]),np.linalg.norm(Wb[i]),np.linalg.norm(W2[:,i]),b1[i],act[i]))
from collections import Counter
print(Counter(r[1] for r in rows if r[7]>0.5))
for r in sorted(rows,key=lambda r:(r[1])):
    print("%3d k=%2d fa=%.2f fb=%.2f fo=%.2f |a|=%.2f |b|=%.2f |o|=%.2f b1=%.2f act=%.2f"%r)
