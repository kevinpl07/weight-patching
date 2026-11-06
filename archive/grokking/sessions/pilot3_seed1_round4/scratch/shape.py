import numpy as np
D="<project>/runs/seed1/loop2/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
alive=np.where(np.abs(W2).sum(0)>0.1)[0]
x=np.arange(p)
def fit(v):
    c=np.fft.fft(v); c[0]=0; k=np.argmax(np.abs(c[1:49]))+1; return k,np.angle(c[k])
print(np.histogram(np.abs(W1[alive]).ravel(),bins=10,range=(0,1.1)))
print(np.histogram(np.abs(W2[:,alive]).ravel(),bins=10,range=(0,1.1)))
# value vs cos
cs=[];vs=[];flips=0;tot=0
for n in alive:
    for v in [W1[n,:p],W1[n,p:],W2[:,n]]:
        k,ph=fit(v); c=np.cos(2*np.pi*k*x/p+ph); cs+=list(c); vs+=list(v)
cs=np.array(cs);vs=np.array(vs)
for lo in np.arange(-1,1,0.2):
    m=(cs>=lo)&(cs<lo+0.2); print("cos[%.1f,%.1f) mean v %.3f  frac sign-disagree %.3f n %d"%(lo,lo+.2,vs[m].mean(),(np.sign(vs[m])!=np.sign(cs[m])).mean(),m.sum()))
