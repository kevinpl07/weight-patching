import numpy as np
S="<project>/runs/seed1/loop/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
alive=[i for i in range(128) if np.linalg.norm(W1[i])>1]
def spec(v):
    F=np.fft.rfft(v); pw=np.abs(F)**2; pw[0]=0; return F,pw/pw.sum()
from collections import Counter
fc=Counter()
for i in alive:
    Fa,pa=spec(W1[i,:p]); Fb,pb=spec(W1[i,p:]); Fo,po=spec(W2[:,i])
    ka,kb,ko=pa.argmax(),pb.argmax(),po.argmax()
    fc[ka]+=1
    ph=np.angle(Fa[ka]),np.angle(Fb[kb]),np.angle(Fo[ko])
    # out phase convention: W2[c,i] ~ cos(w c - phi_o), W1a ~ cos(w a + phi_a)? rfft gives sum v e^{-iwk}, so v~cos(wk+angle)
    print(i,ka,kb,ko,"pow %.2f %.2f %.2f"%(pa[ka],pb[kb],po[ko]),"b1 %.3f"%b1[i],
          "phase a+b-o %.2f"%(((ph[0]+ph[1]-ph[2])+np.pi)%(2*np.pi)-np.pi))
print(sorted(fc.items()))
