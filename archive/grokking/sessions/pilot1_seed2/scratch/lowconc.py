import numpy as np
D=r"<project>\runs\seed2\agent"
W=dict(np.load(D+r"\weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
k,conc=np.load(D+r"\scratch\conc.npy"); k=k.astype(int)
Fa=np.fft.rfft(w1[:,:p],axis=1); Fb=np.fft.rfft(w1[:,p:],axis=1); Fo=np.fft.rfft(w2.T,axis=1)
Pa,Pb,Po=[np.abs(F[:,1:])**2 for F in (Fa,Fb,Fo)]
ka,kb,ko=Pa.argmax(1)+1,Pb.argmax(1)+1,Po.argmax(1)+1
mm=np.array([(np.angle(Fo[n,k[n]])-np.angle(Fa[n,k[n]])-np.angle(Fb[n,k[n]])+np.pi)%(2*np.pi)-np.pi for n in range(128)])
for n in np.argsort(conc):
    if conc[n]>0.25: break
    print(n,round(conc[n],3),k[n],(ka[n],kb[n],ko[n]),"mm",round(mm[n],2),"bias",round(b1[n],2))
print("b2",np.round(b2,2)); 
T=np.load(D+r"\train_pairs.npy"); cnt=np.bincount(T[:,2],minlength=p)
print("corr b2 vs label count",np.corrcoef(b2,cnt)[0,1])
