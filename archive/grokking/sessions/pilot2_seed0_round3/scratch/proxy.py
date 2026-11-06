import numpy as np
D="<project>/runs/seed0/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); p=97
tr=np.load(D+"train_pairs.npy"); A_,B_,Y=tr[:,0],tr[:,1],tr[:,2]
x=np.arange(p)
def fwd(W):
    h=np.maximum(W["fc1.weight"][:,A_]+W["fc1.weight"][:,p+B_]+W["fc1.bias"][:,None],0).T
    L=h@W["fc2.weight"].T+W["fc2.bias"]
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(Y)),Y].mean()),4), round(float((L.argmax(1)==Y).mean()),4)
def basis(ks):
    B=[np.ones(p)]
    for k in ks: B+=[np.cos(2*np.pi*k*x/p),np.sin(2*np.pi*k*x/p)]
    Q,_=np.linalg.qr(np.array(B).T); return Q
def proj(v,ks): Q=basis(ks); return Q@(Q.T@v)
W1=W["fc1.weight"]; live=[n for n in range(128) if np.linalg.norm(W1[n])>1]
dead=[n for n in range(128) if n not in live]
def kof(n): P=np.abs(np.fft.rfft(W1[n,:p]))**2; return np.argmax(P[1:])+1
H=lambda k:[k,(2*k)%p,(3*k)%p]
P={k:v.copy() for k,v in W.items()}
for n in live:
    P["fc1.weight"][n,:p]=proj(W1[n,:p],H(kof(n))); P["fc1.weight"][n,p:]=proj(W1[n,p:],H(kof(n)))
    P["fc2.weight"][:,n]=proj(W["fc2.weight"][:,n],H(kof(n)))
singles=[39,34,18,41,54,22,115,78]
print("orig",fwd(W),"proj",fwd(P))
def kill(V,ns):
    V={k:v.copy() for k,v in V.items()}
    for n in ns: V["fc1.bias"][n]=-100
    return V
def partner(V,ns):
    V={k:v.copy() for k,v in V.items()}
    for i,n in enumerate(ns):
        m=dead[i]
        V["fc1.weight"][m]=-V["fc1.weight"][n]; V["fc1.bias"][m]=V["fc1.bias"][n]; V["fc2.weight"][:,m]=V["fc2.weight"][:,n]
    return V
for nm,f in [("kill",kill),("partner",partner)]:
    print(nm,"proj:",fwd(f(P,singles)),"orig:",fwd(f(W,singles)))
pairs=[80,82,42,108,14,35,0,87,3,91,23,33,17,72,63,84,24,122,74,123]
print("partner all 2-groups+singles proj", fwd(partner(P,singles+pairs)))
