import numpy as np, sys
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz"))
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
Wa=W1[:,:p]; Wb=W1[:,p:]
def four(v):
    F=np.fft.rfft(v - v.mean())
    pw=np.abs(F)**2
    k=np.argmax(pw[1:])+1
    return k, pw[k]/pw[1:].sum(), np.angle(F[k]), np.abs(F[k])
tr=np.load(S+"train_pairs.npy")
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),tr[:,0]]=1; X[np.arange(len(tr)),p+tr[:,1]]=1
H=np.maximum(X@W1.T+b1,0)
act=(H>0).mean(0)
rows=[]
for n in range(128):
    ka,ca,pa,ma=four(Wa[n]); kb,cb,pb,mb=four(Wb[n]); ko,co,po,mo=four(W2[:,n])
    rows.append((n,ka,kb,ko,ca,cb,co,ma,mb,mo,b1[n],act[n],np.linalg.norm(W2[:,n]), Wa[n].mean(),Wb[n].mean()))
    print(f"{n:3d} k=({ka:2d},{kb:2d},{ko:2d}) conc=({ca:.2f},{cb:.2f},{co:.2f}) amp=({ma:.1f},{mb:.1f},{mo:.1f}) b={b1[n]:+.2f} act={act[n]:.2f} |w2|={np.linalg.norm(W2[:,n]):.2f} meanA={Wa[n].mean():+.2f} meanB={Wb[n].mean():+.2f} dph={(po-pa-pb+np.pi)%(2*np.pi)-np.pi:+.2f}")
