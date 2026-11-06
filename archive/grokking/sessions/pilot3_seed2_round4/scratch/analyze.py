import numpy as np, sys
S="<project>/runs/seed2/loop2/round4/agent/"
W=dict(np.load(S+(sys.argv[1] if len(sys.argv)>1 else "weights.npz")))
p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
A=w1[:,:p];B=w1[:,p:]
def four(v):
    F=np.fft.rfft(v); pw=np.abs(F)**2; pw[0]=0
    k=np.argmax(pw); return k, pw[k]/pw.sum(), np.angle(F[k]), np.abs(F[k])
d=np.load(S+"train_pairs.npy")
h=np.maximum(0, A.T[d[:,0]]+B.T[d[:,1]]+b1)
print("i  |a| |b| bias |out| act | fa ea pa | fb eb pb | fo eo")
rows=[]
for i in range(128):
    na,nb,no=np.linalg.norm(A[i]),np.linalg.norm(B[i]),np.linalg.norm(w2[:,i])
    fa,ea,pa,_=four(A[i]); fb,eb,pb,_=four(B[i]); fo,eo,po,_=four(w2[:,i])
    act=(h[:,i]>0).mean()
    rows.append((i,na,nb,b1[i],no,act,fa,ea,pa,fb,eb,pb,fo,eo,po))
    print(f"{i:3d} {na:5.2f} {nb:5.2f} {b1[i]:+.2f} {no:5.2f} {act:.2f} | {fa:2d} {ea:.2f} {pa:+.2f} | {fb:2d} {eb:.2f} {pb:+.2f} | {fo:2d} {eo:.2f} {po:+.2f}")
np.save(S+"scratch/rows.npy",np.array(rows))
