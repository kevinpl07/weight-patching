import numpy as np
# Fourier coefficients of ReLU(cos u + cos v + t) on a continuous grid (pure math, no data)
n=256; u=np.linspace(0,2*np.pi,n,endpoint=False); U,V=np.meshgrid(u,u,indexing='ij')
for t in [0.6,0.3,0,-0.3,-0.6,-0.9,-1.2]:
    H=np.maximum(np.cos(U)+np.cos(V)+t,0)
    F=np.fft.fft2(H)/n/n
    sig=abs(F[1,1]); lin=abs(F[1,0]); amb=abs(F[1,-1]); h2=abs(F[2,0]); dc=abs(F[0,0])
    print("t %+.1f  sig(u+v) %.3f lin %.3f a-b %.3f 2u %.3f dc %.3f  sig/lin %.2f"%(t,sig,lin,amb,h2,dc,sig/lin))
