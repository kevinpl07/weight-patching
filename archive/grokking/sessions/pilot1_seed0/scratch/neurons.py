import numpy as np
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
p=97
W1=W["fc1.weight"]; Wa=W1[:,:p]; Wb=W1[:,p:]; W2=W["fc2.weight"]
Fa=np.fft.fft(Wa,axis=1); Fb=np.fft.fft(Wb,axis=1); Fo=np.fft.fft(W2.T,axis=1)
def top(F):
    P=np.abs(F[:,1:49])**2; k=P.argmax(1)+1; fr=P.max(1)/P.sum(1); return k,fr
ka,fa=top(Fa); kb,fb=top(Fb); ko,fo=top(Fo)
rows=[]
for i in range(128):
    k=ka[i]
    # phase relation: out freq k phase should be conj? check angle(Fo[k]) vs angle(Fa)+angle(Fb)
    ph=np.angle(Fa[i,k])+np.angle(Fb[i,k])-np.angle(Fo[i,k])
    ph=(ph+np.pi)%(2*np.pi)-np.pi
    rows.append((i,ka[i],fa[i],kb[i],fb[i],ko[i],fo[i],ph))
rows.sort(key=lambda r:-(r[2]+r[4]+r[6]))
for r in rows: print("%3d | a k=%2d %.2f | b k=%2d %.2f | o k=%2d %.2f | dphase %.2f"%r)
from collections import Counter
print(Counter([r[1] for r in rows if r[1]==r[3]==r[5]]))
