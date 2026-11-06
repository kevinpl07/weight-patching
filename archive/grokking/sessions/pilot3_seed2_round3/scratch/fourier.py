import numpy as np
W=dict(np.load('weights.npz'))
p=97
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
A=W1[:,:p]; B=W1[:,p:]
live=[n for n in range(128) if np.linalg.norm(A[n])>1]
print('live',len(live))
def spec(x):
    f=np.fft.rfft(x - x.mean()); e=np.abs(f)**2; return f,e
rows=[]
for n in live:
    fa,ea=spec(A[n]); fb,eb=spec(B[n]); fo,eo=spec(W2[:,n])
    ka=ea.argmax(); kb=eb.argmax(); ko=eo.argmax()
    fr=lambda e,k: e[k]/e.sum()
    pa=np.angle(fa[ka]); pb=np.angle(fb[kb]); po=np.angle(fo[ko])
    rows.append((n,ka,kb,ko,fr(ea,ka),fr(eb,kb),fr(eo,ko),np.angle(np.exp(1j*(pa+pb-po))), A[n].mean(),B[n].mean(),W2[:,n].mean(), b1[n]))
rows.sort(key=lambda r:r[1])
for r in rows: print('n%3d ka%2d kb%2d ko%2d fa%.2f fb%.2f fo%.2f dphi%+.2f meanA%+.2f meanB%+.2f meanO%+.2f b%+.3f'%r)
from collections import Counter
print(Counter([r[1] for r in rows]))
fb2=np.fft.rfft(b2-b2.mean()); e=np.abs(fb2)**2; print('b2 top freq',np.argsort(e)[::-1][:5], (np.sort(e)[::-1][:5]/e.sum()).round(3))
