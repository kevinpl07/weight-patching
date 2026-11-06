import numpy as np, sys
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+(sys.argv[1] if len(sys.argv)>1 else "weights.npz")))
p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
A=W1[:,:p]; B=W1[:,p:]; O=W2.T  # 128x97
def spec(X):
    F=np.fft.rfft(X,axis=1); P=np.abs(F)**2; P[:,0]=0
    return F,P
FA,PA=spec(A);FB,PB=spec(B);FO,PO=spec(O)
tot=PA+PB+PO
for n in range(128):
    f=np.argmax(tot[n]); 
    ca=PA[n,f]/PA[n].sum(); cb=PB[n,f]/PB[n].sum(); co=PO[n,f]/PO[n].sum()
    pa=np.angle(FA[n,f]);pb=np.angle(FB[n,f]);po=np.angle(FO[n,f])
    # phase relation: out ~ cos(w(c)+..). check po vs -(pa+pb)? print residual
    r1=np.angle(np.exp(1j*(po-(pa+pb)))); 
    dcA=A[n].mean();dcB=B[n].mean()
    print(f"{n:3d} f={f:2d} conc a{ca:.2f} b{cb:.2f} o{co:.2f}  dphase {r1:+.2f} dcA {dcA:+.2f} dcB {dcB:+.2f} bias {b1[n]:+.3f}")
