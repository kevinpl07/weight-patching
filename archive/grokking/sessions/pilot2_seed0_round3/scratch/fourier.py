import numpy as np
D="<project>/runs/seed0/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
Wa=W1[:,:p]; Wb=W1[:,p:]
def spec(v):  # power per freq 0..48
    F=np.fft.rfft(v); P=np.abs(F)**2; return P
tr=np.load(D+"train_pairs.npy")
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),tr[:,0]]=1; X[np.arange(len(tr)),p+tr[:,1]]=1
H=np.maximum(X@W1.T+b1,0)
act=(H>0).mean(0)
for n in range(128):
    Pa=spec(Wa[n]); Pb=spec(Wb[n]); Po=spec(W2[:,n])
    ka=np.argmax(Pa[1:])+1; kb=np.argmax(Pb[1:])+1; ko=np.argmax(Po[1:])+1
    pa=Pa[ka]/Pa[1:].sum(); pb=Pb[kb]/Pb[1:].sum(); po=Po[ko]/Po[1:].sum()
    print(f"{n:3d} ka{ka:2d} {pa:.2f} kb{kb:2d} {pb:.2f} ko{ko:2d} {po:.2f} |a|{np.linalg.norm(Wa[n]):.2f} |b|{np.linalg.norm(Wb[n]):.2f} |o|{np.linalg.norm(W2[:,n]):.2f} bias{b1[n]:.2f} act{act[n]:.2f} mean_a{Wa[n].mean():.2f}")
