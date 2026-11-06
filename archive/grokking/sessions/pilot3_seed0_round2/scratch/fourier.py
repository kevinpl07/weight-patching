import numpy as np, sys
S="<project>/runs/seed0/loop2/round2/agent/"
W=dict(np.load(S+(sys.argv[1] if len(sys.argv)>1 else "weights.npz")))
p=97
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
Wa=W1[:,:p]; Wb=W1[:,p:]
live=np.where(np.linalg.norm(Wa,axis=1)>1)[0]
print("live",len(live))
def spec(v):
    F=np.fft.rfft(v); P=np.abs(F)**2; P[0]=0; return P/P.sum()
rows=[]
for n in live:
    sa=spec(Wa[n]); sb=spec(Wb[n]); so=spec(W2[:,n])
    ka,kb,ko=sa.argmax(),sb.argmax(),so.argmax()
    rows.append((n,ka,sa[ka],kb,sb[kb],ko,so[ko], Wa[n].mean(),Wb[n].mean(),W2[:,n].mean()))
    print(f"{n:4d} a:k={ka:2d} {sa[ka]:.2f}  b:k={kb:2d} {sb[kb]:.2f}  out:k={ko:2d} {so[ko]:.2f}  means {Wa[n].mean():+.3f} {Wb[n].mean():+.3f} {W2[:,n].mean():+.3f}")
from collections import Counter
print(Counter(r[1] for r in rows))
# bias spectrum
print("b2 spec top:", np.argsort(spec(b2))[::-1][:5], np.sort(spec(b2))[::-1][:5], "b2 mean",b2.mean())
# overall freq power in Wa (live)
P=sum(spec(Wa[n])*np.linalg.norm(Wa[n])**2 for n in live)
print("Wa total power by k top:", [(k,round(P[k]/P.sum(),3)) for k in np.argsort(P)[::-1][:10]])
P=sum(spec(W2[:,n])*np.linalg.norm(W2[:,n])**2 for n in live)
print("W2 total power by k top:", [(k,round(P[k]/P.sum(),3)) for k in np.argsort(P)[::-1][:10]])
