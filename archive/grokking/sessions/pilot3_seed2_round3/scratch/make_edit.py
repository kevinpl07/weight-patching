# Build phase-shifted (a-phase +pi/2, b-phase -pi/2) sparse seeds on dead neurons.
# Keeps a+b phase (so same output pattern) but flips a-b phase by pi -> cancels a-b cross term.
import numpy as np, json, sys
nper=int(sys.argv[1]); out=sys.argv[2]; freqs=[int(x) for x in sys.argv[3].split(',')]
W=dict(np.load('weights.npz')); p=97
W1=W['fc1.weight']; A=W1[:,:p]; B=W1[:,p:]
live=[n for n in range(128) if np.linalg.norm(A[n])>1]
dead=[n for n in range(128) if n not in live and n not in (4,6,100)]
src={}
for n in live:
    fa=np.fft.rfft(A[n]); k=np.abs(fa[1:]).argmax()+1
    src.setdefault(k,[]).append(n)
x=np.arange(p); ops=[]; used=0
for i,k in enumerate(freqs):
    n=src[k][0]; t=dead[i]
    fa=np.fft.rfft(A[n]); fb=np.fft.rfft(B[n])
    pa=np.angle(fa[k])+np.pi/2; pb=np.angle(fb[k])-np.pi/2
    ca=np.cos(2*np.pi*k*x/p+pa); cb=np.cos(2*np.pi*k*x/p+pb)
    h=nper//2
    ia=np.argsort(-np.abs(ca))[:h]; ib=np.argsort(-np.abs(cb))[:nper-h]
    cols=list(map(int,ia))+[int(j)+p for j in ib]
    vals=[float(ca[j]) for j in ia]+[float(cb[j]) for j in ib]
    ops.append({"tensor":"fc1.weight","rows":[t],"cols":cols,"op":"set","value":[vals]})
    ops.append({"tensor":"fc1.weight","rows":[t],"cols":cols,"op":"reset_optimizer"})
    used+=len(cols); print('k',k,'src',n,'-> dead',t)
print('entries',used)
json.dump(ops,open(out,'w'))
