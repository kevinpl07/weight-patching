import numpy as np
D=r"<project>\runs\seed2\agent"
W=dict(np.load(D+r"\weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
k,conc=np.load(D+r"\scratch\conc.npy"); k=k.astype(int)
Fa=np.fft.rfft(w1[:,:p],axis=1); Fb=np.fft.rfft(w1[:,p:],axis=1); Fo=np.fft.rfft(w2.T,axis=1)
order=np.argsort(-conc)
for n in order[:60]:
    kk=k[n]; pa,pb,po=np.angle(Fa[n,kk]),np.angle(Fb[n,kk]),np.angle(Fo[n,kk])
    d=(po-pa-pb+np.pi)%(2*np.pi)-np.pi
    amp=lambda F:2*np.abs(F[n,kk])/p
    print(n,kk,round(conc[n],2),"amp a/b/o",round(amp(Fa),2),round(amp(Fb),2),round(amp(Fo),2),"res a/b/o",
      round(np.sqrt((np.abs(Fa[n,1:])**2).sum()-np.abs(Fa[n,kk])**2)*np.sqrt(2)/p,2),
      round(np.sqrt((np.abs(Fo[n,1:])**2).sum()-np.abs(Fo[n,kk])**2)*np.sqrt(2)/p,2),
      "phase a,b,o",round(pa,2),round(pb,2),round(po,2),"mismatch",round(d,2),"dc a,b",round(w1[n,:p].mean(),2),round(w1[n,p:].mean(),2),"bias",round(b1[n],2))
