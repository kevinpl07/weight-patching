import numpy as np
p=97
W=dict(np.load("weights.npz"))
W1=W["fc1.weight"];b1=W["fc1.bias"];W2=W["fc2.weight"];b2=W["fc2.bias"]
live=[i for i in range(128) if np.linalg.norm(W1[i])>1]
print("live",len(live))
def spec(v):
    F=np.fft.rfft(v-v.mean()); e=np.abs(F)**2; e[0]=0
    return e/e.sum(), F
rows=[]
for i in live:
    ea,Fa=spec(W1[i,:p]); eb,Fb=spec(W1[i,p:]); eo,Fo=spec(W2[:,i])
    fa,fb,fo=ea.argmax(),eb.argmax(),eo.argmax()
    rows.append((i,fa,ea[fa],fb,eb[fb],fo,eo[fo], W1[i,:p].mean(), W1[i,p:].mean(), W2[:,i].mean()))
from collections import Counter
print(Counter(r[1] for r in rows))
for r in rows: print("%3d fa=%2d %.2f fb=%2d %.2f fo=%2d %.2f  means %.3f %.3f %.3f"%r)
print("b2 stats",b2.mean(),b2.std())
print("b2 spectrum", np.round(spec(b2)[0],3).tolist())
