import numpy as np
p=97
W=dict(np.load('weights.npz'))
w1,b1,w2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
live=[i for i in range(128) if np.linalg.norm(w1[i])>0.01]
def fit(v,k):
    F=np.fft.fft(v); c=F[k]*2/p; return np.abs(c), np.angle(c)
res={}
for i in live:
    sa=np.abs(np.fft.rfft(w1[i,:p]))[1:]; k=sa.argmax()+1
    _,pa=fit(w1[i,:p],k); _,pb=fit(w1[i,p:],k); _,po=fit(w2[:,i],k)
    res.setdefault(k,[]).append((i,pa%(2*np.pi),pb%(2*np.pi),po%(2*np.pi)))
for k in sorted(res):
    print(k, ' | '.join("n%d a%.2f b%.2f o%.2f"%t for t in res[k]))
