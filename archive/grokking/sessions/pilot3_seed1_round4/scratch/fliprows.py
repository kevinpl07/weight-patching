import numpy as np
D="<project>/runs/seed1/loop2/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97; x=np.arange(p)
T=np.load(D+"train_pairs.npy"); a,b,y=T[:,0],T[:,1],T[:,2]
W1,b1,W2,b2=[W[k] for k in ['fc1.weight','fc1.bias','fc2.weight','fc2.bias']]
alive=np.where(np.abs(W2).sum(0)>0.1)[0]
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); mh=h.mean(0)
def fitcos(v):
    c=np.fft.fft(v); c[0]=0; k=np.argmax(np.abs(c[1:49]))+1; return np.cos(2*np.pi*k*x/p+np.angle(c[k]))
cnt=np.zeros(p); eff=np.zeros(p)
for n in alive:
    c=fitcos(W2[:,n]); t=np.clip(1.3*c,-.88,.88)
    m=(np.sign(W2[:,n])!=np.sign(c))&(np.abs(c)>0.08)
    cnt+=m; eff+=m*(W2[:,n]-t)*mh[n]
o=np.argsort(eff)
print("rows by offset contribution of flips (low):",[(int(i),int(cnt[i]),round(eff[i],2),round(float(b2[i]),2)) for i in o[:8]])
print("high:",[(int(i),int(cnt[i]),round(eff[i],2),round(float(b2[i]),2)) for i in o[-8:]])
print("count dist",np.bincount(cnt.astype(int)))
print("corr eff vs b2",np.corrcoef(eff,b2)[0,1])
