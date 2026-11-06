import numpy as np
S="<project>/runs/seed0/loop2/round2/agent/"
W0=dict(np.load(S+"weights.npz")); p=97; z=np.load(S+"scratch/optim0.npz")
U1=z['next_update/fc1.weight']; U2=z['next_update/fc2.weight']
G1=-z['grad/fc1.weight']; G2=-z['grad/fc2.weight']
def parts(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]; s=np.fft.irfft(G,n=p); return s, v-s
live=np.where(np.linalg.norm(W0['fc1.weight'][:,:p],axis=1)>1)[0]
tot=dict(sig_u=0,noi_u=0,sig_g=0,noi_g=0)
for n in live:
    P=np.abs(np.fft.rfft(W0['fc1.weight'][n,:p]))**2; P[0]=0; k=int(P.argmax())
    for v,u,g in ((W0['fc1.weight'][n,:p],U1[n,:p],G1[n,:p]),(W0['fc1.weight'][n,p:],U1[n,p:],G1[n,p:]),(W0['fc2.weight'][:,n],U2[:,n],G2[:,n])):
        s,e=parts(v,k)
        tot['sig_u']+=s@u/np.linalg.norm(s); tot['noi_u']+=e@u/np.linalg.norm(e)
        tot['sig_g']+=s@g/np.linalg.norm(s); tot['noi_g']+=e@g/np.linalg.norm(e)
print({k:round(float(v),5) for k,v in tot.items()})
