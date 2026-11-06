import numpy as np
W=dict(np.load('weights.npz'))
p=97
W1=W['fc1.weight']; W2=W['fc2.weight']
A=W1[:,:p]; B=W1[:,p:]
live=[n for n in range(128) if np.linalg.norm(A[n])>1]
byk={}
for n in live:
    fa=np.fft.rfft(A[n]); k=np.abs(fa[1:]).argmax()+1
    fb=np.fft.rfft(B[n]); fo=np.fft.rfft(W2[:,n])
    byk.setdefault(k,[]).append((n,np.angle(fa[k]),np.angle(fb[k]),np.angle(fo[k]),abs(fa[k]),abs(fb[k]),abs(fo[k])))
for k in sorted(byk):
    s=' | '.join('n%d d=%+.2f amp %.0f/%.0f/%.0f'%(n,np.angle(np.exp(1j*(pa-pb))),aa,ab,ao) for n,pa,pb,po,aa,ab,ao in byk[k])
    print(k,s)
