import numpy as np
p=97
W=dict(np.load('weights.npz'))
w1,b1,w2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
dead=[i for i in range(128) if np.linalg.norm(w1[i])<0.01]
print('dead',len(dead),'max abs dead w1',np.abs(w1[dead]).max(),'b1',np.abs(b1[dead]).max(),'w2',np.abs(w2[:,dead]).max())
live=[i for i in range(128) if i not in dead]
x=np.arange(p)
def fit(v,k):
    F=np.fft.fft(v); c=F[k]*2/p
    return np.abs(c), np.angle(c), v.mean()  # v ~ amp cos(2pi k x/p + phase)
rows=[]
for i in live:
    sa=np.abs(np.fft.rfft(w1[i,:p]))[1:]; k=sa.argmax()+1
    if k>48: k=p-k
    aa,pa,ma=fit(w1[i,:p],k); ab,pb,mb=fit(w1[i,p:],k); ao,po,mo=fit(w2[:,i],k)
    d=(po-(pa+pb)+np.pi)%(2*np.pi)-np.pi
    dd=(pa-pb)%(2*np.pi)
    rows.append((k,i,aa,ab,ao,ma+mb+b1[i],d,dd))
rows.sort()
for r in rows: print("k %2d n %3d ampA %.2f ampB %.2f ampO %.2f const %+.2f  psi-(pa+pb) %+.2f  pa-pb %.2f"%r)
