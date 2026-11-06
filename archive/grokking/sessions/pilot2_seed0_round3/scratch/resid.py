import numpy as np
D="<project>/runs/seed0/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
live=[n for n in range(128) if np.linalg.norm(W1[n])>1]
print(len(live))
x=np.arange(p)
def proj(v,ks,const=True):
    B=[np.ones(p)] if const else []
    for k in ks: B+= [np.cos(2*np.pi*k*x/p), np.sin(2*np.pi*k*x/p)]
    B=np.array(B).T
    c=np.linalg.lstsq(B,v,rcond=None)[0]   # projection onto chosen basis (orthogonal), not fitting to data
    return B@c
tot={'a':[],'b':[],'o':[]}
for n in live:
    for nm,v in [('a',W1[n,:p]),('b',W1[n,p:]),('o',W2[:,n])]:
        P=np.abs(np.fft.rfft(v))**2; k=np.argmax(P[1:])+1
        r=v-proj(v,[k])
        s=np.sort(r**2)[::-1]
        Pr=np.abs(np.fft.rfft(r))**2
        topf=np.argsort(Pr)[::-1][:3]
        tot[nm].append((n,k,np.sum(r**2)/np.sum(v**2), s[:10].sum()/s.sum(), list(topf), np.round(Pr[topf]/Pr.sum(),2)))
for nm in 'abo':
    print(nm)
    for t in tot[nm][:12]: print(t)
    print('mean resid frac',np.mean([t[2] for t in tot[nm]]),'top10 share',np.mean([t[3] for t in tot[nm]]))
