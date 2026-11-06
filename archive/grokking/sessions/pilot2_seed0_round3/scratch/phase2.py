import numpy as np
D="<project>/runs/seed0/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1=W["fc1.weight"]; W2=W["fc2.weight"]
live=[n for n in range(128) if np.linalg.norm(W1[n])>1]
x=np.arange(p)
def fc(v,k):
    c=np.sum(v*np.exp(-2j*np.pi*k*x/p))*2/p; return c
def kof(n): P=np.abs(np.fft.rfft(W1[n,:p]))**2; return np.argmax(P[1:])+1
g={}
for n in live: g.setdefault(kof(n),[]).append(n)
for k,ns in sorted(g.items()):
    s=[]
    for n in ns:
        a=fc(W1[n,:p],k); b=fc(W1[n,p:],k); o=fc(W2[:,n],k)
        s.append(f"n{n}: al{np.degrees(np.angle(a)):+5.0f} be{np.degrees(np.angle(b)):+5.0f} ga{np.degrees(np.angle(o)):+5.0f}")
    print(k," | ".join(s))
