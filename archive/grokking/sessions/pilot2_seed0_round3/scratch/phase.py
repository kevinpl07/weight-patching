import numpy as np
D="<project>/runs/seed0/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
live=[n for n in range(128) if np.linalg.norm(W1[n])>1]
x=np.arange(p)
def fc(v,k):
    c=np.sum(v*np.exp(-2j*np.pi*k*x/p))*2/p   # v ≈ |c| cos(kx+arg c)
    return np.abs(c), np.angle(c)
rows=[]
for n in live:
    P=np.abs(np.fft.rfft(W1[n,:p]))**2; k=np.argmax(P[1:])+1
    A,al=fc(W1[n,:p],k); B,be=fc(W1[n,p:],k); C,ga=fc(W2[:,n],k)
    err=np.angle(np.exp(1j*(ga-al-be)))
    rows.append((k,n,A,B,C,err,W1[n,:p].mean(),W1[n,p:].mean(),b1[n]))
rows.sort()
for r in rows: print("k%2d n%3d A%.2f B%.2f C%.2f phase_err%+.2f mean_a%+.2f mean_b%+.2f bias%+.2f"%r)
