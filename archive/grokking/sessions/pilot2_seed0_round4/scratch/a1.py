import numpy as np, sys
p=97
W=dict(np.load('weights.npz'))
w1,b1,w2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
d=np.load('train_pairs.npy'); A,B,Y=d[:,0],d[:,1],d[:,2]
H=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0)
L=H@w2.T+b2
print('acc',(L.argmax(1)==Y).mean())
def spec(v):
    F=np.fft.rfft(v); e=np.abs(F[1:])**2; return e/e.sum()
for i in range(128):
    sa=spec(w1[i,:p]); sb=spec(w1[i,p:]); so=spec(w2[:,i])
    ka=sa.argmax()+1; kb=sb.argmax()+1; ko=so.argmax()+1
    print(i, f"na {np.linalg.norm(w1[i,:p]):.2f} nb {np.linalg.norm(w1[i,p:]):.2f} b {b1[i]:+.2f} no {np.linalg.norm(w2[:,i]):.2f} act {(H[:,i]>0).mean():.2f}", ka, f"{sa.max():.2f}", kb, f"{sb.max():.2f}", ko, f"{so.max():.2f}")
