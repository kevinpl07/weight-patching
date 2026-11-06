import numpy as np
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
A=W1[:,:p];B=W1[:,p:];O=W2.T
def fold(k): k%=p; return min(k,p-k)
for name,X in [("A",A),("B",B),("O",O)]:
    F=np.fft.rfft(X,axis=1);P=np.abs(F)**2;P[:,0]=0
    f=P.argmax(1)
    res=[]
    for n in range(128):
        if P[n,f[n]]/P[n].sum()<0.6: continue
        tot=P[n].sum(); h={k:P[n,fold(k*f[n])]/tot for k in [1,2,3,5,7]}
        rest=1-h[1]
        res.append((h[1],h[2],h[3],h[5],h[7],rest))
    r=np.array(res); print(name,"mean frac f,2f,3f,5f,7f, offtotal",r.mean(0).round(3))
# value distribution of clean neuron weights vs cos fit
F=np.fft.rfft(A,axis=1);P=np.abs(F)**2;P[:,0]=0;f=P.argmax(1)
n=0; ph=np.angle(F[n,f[n]]); amp=2*np.abs(F[n,f[n]])/p
x=np.arange(p); c=amp*np.cos(2*np.pi*f[n]*x/p+ph)
print("neuron0 amp",amp); idx=np.argsort(c)
print(np.c_[c[idx],A[n,idx]][::6].round(2))
