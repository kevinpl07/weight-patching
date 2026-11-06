import numpy as np
S="<project>/runs/seed0/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W['fc1.weight']; Wa=W1[:,:p]; Wb=W1[:,p:]; W2=W['fc2.weight']; b1=W['fc1.bias']
live=np.where(np.linalg.norm(Wa,axis=1)>1)[0]
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[0]=F[0]; G[k]=F[k]; return np.fft.irfft(G,n=p)
def phase(v,k): return np.angle(np.fft.rfft(v)[k])
tot=0
for n in live:
    P=np.abs(np.fft.rfft(Wa[n]))**2; P[0]=0
    k=int(np.argmax(P+np.abs(np.fft.rfft(Wb[n]))**2*(np.arange(49)>0)))
    na=np.linalg.norm(Wa[n]-proj(Wa[n],k)); nb=np.linalg.norm(Wb[n]-proj(Wb[n],k)); no=np.linalg.norm(W2[:,n]-proj(W2[:,n],k))
    pa,pb,po=phase(Wa[n],k),phase(Wb[n],k),phase(W2[:,n],k)
    d=(po-pa-pb+np.pi)%(2*np.pi)-np.pi
    dab=(pa-pb+np.pi)%(2*np.pi)-np.pi
    ampa=2*np.abs(np.fft.rfft(Wa[n])[k])/p; ampb=2*np.abs(np.fft.rfft(Wb[n])[k])/p; ampo=2*np.abs(np.fft.rfft(W2[:,n])[k])/p
    print(f"{n:4d} k={k:2d} noise a {na:.2f} b {nb:.2f} o {no:.2f} | amp {ampa:.2f} {ampb:.2f} {ampo:.2f} | psi-pa-pb {d:+.2f} pa-pb {dab:+.2f} b1 {b1[n]:+.3f}")
