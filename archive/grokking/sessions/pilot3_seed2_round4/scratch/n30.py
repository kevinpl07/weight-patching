import numpy as np
S="<project>/runs/seed2/loop2/round4/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
A=w1[:,:p];B=w1[:,p:]
for i in [30,91,96]:
    for nm,v in [("a",A[i]),("b",B[i]),("o",w2[:,i])]:
        F=np.fft.rfft(v); m=np.abs(F); top=np.argsort(-m[1:])[:6]+1
        print(i,nm,"mean %.3f"%v.mean()," ".join(f"{k}:{m[k]:.1f}@{np.angle(F[k]):+.2f}" for k in top))
# continuous Fourier coefs of relu(cos ta + cos tb)
t=np.linspace(0,2*np.pi,400,endpoint=False); TA,TB=np.meshgrid(t,t,indexing='ij')
R=np.maximum(0,np.cos(TA)+np.cos(TB)); F=np.fft.fft2(R)/R.size
for kk in [(0,0),(1,0),(1,1),(1,-1),(2,0),(2,1),(2,2),(3,0),(2,-2)]: print(kk, abs(F[kk]).round(4))
print("b2 stats",b2.mean(),b2.std())
