import numpy as np
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
Wa=W1[:,:p]; Wb=W1[:,p:]
def purity(M):
    P=np.abs(np.fft.fft(M,axis=1)[:,1:49])**2; return P.max(1)/P.sum(1), P.argmax(1)+1
pa,ka=purity(Wa); pb,kb=purity(Wb); po,ko=purity(W2.T)
pur=np.minimum(np.minimum(pa,pb),po); same=(ka==kb)&(kb==ko)
def run(mask, b2v=None, scale_out=1.0):
    h=np.maximum(Wa[:,A].T+Wb[:,B].T+b1,0)*mask
    lg=h@W2.T*scale_out+(b2 if b2v is None else b2v)
    lg=lg-lg.max(1,keepdims=True); lp=lg-np.log(np.exp(lg).sum(1,keepdims=True))
    return (lg.argmax(1)==Y).mean(), -lp[np.arange(len(Y)),Y].mean()
print("orig",run(np.ones(128)))
for T in [0.6,0.5,0.4,0.3,0.2,0.15,0.12]:
    m=(same&(pur>=T)).astype(float)
    print(T,int(m.sum()),"keep", run(m), "b2=0", run(m,np.zeros(p)), "b2=0 x3", run(m,np.zeros(p),3.0))
print("---- abs activation (as if negated partners existed)")
def run_abs(mask):
    X=Wa[:,A].T+Wb[:,B].T+b1
    h=np.where(mask>0, np.abs(X)/2, 0)  # relu(x)+relu(-x) /2 normalization
    lg=h@W2.T
    lg=lg-lg.max(1,keepdims=True); lp=lg-np.log(np.exp(lg).sum(1,keepdims=True))
    return (lg.argmax(1)==Y).mean(), -lp[np.arange(len(Y)),Y].mean()
for T in [0.5,0.3,0.2,0.12]:
    m=(same&(pur>=T)).astype(float); print(T,int(m.sum()),run_abs(m))
# phases per frequency for pure neurons
m=same&(pur>=0.3)
Fa=np.fft.fft(Wa,axis=1); Fb=np.fft.fft(Wb,axis=1); Fo=np.fft.fft(W2.T,axis=1)
for k in sorted(set(ka[m])):
    idx=np.where(m&(ka==k))[0]
    print(k,[(int(i),round(float(np.angle(Fa[i,k])),2),round(float(np.angle(Fb[i,k])),2),round(float(np.angle(Fo[i,k])),2), round(float(b1[i]),2)) for i in idx])
