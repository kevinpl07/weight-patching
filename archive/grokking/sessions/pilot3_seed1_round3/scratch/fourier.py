import numpy as np, sys
S="<project>/runs/seed1/loop2/round3/agent/"
W=dict(np.load(S+(sys.argv[1] if len(sys.argv)>1 else "weights.npz")))
p=97
W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']
A=W1[:,:p]; B=W1[:,p:]; O=W2.T  # O: 128 x 97
alive=np.where(np.linalg.norm(A,axis=1)>1)[0]
def spec(v):
    F=np.fft.rfft(v-v.mean()); pw=np.abs(F[1:])**2; return pw/pw.sum(), np.angle(F[1:])
rows=[]
from collections import Counter
cnt=Counter()
for n in alive:
    sa,pa=spec(A[n]); sb,pb=spec(B[n]); so,po=spec(O[n])
    tot=sa+sb+so; k=np.argmax(tot)+1
    fa,fb,fo=sa[k-1],sb[k-1],so[k-1]
    # out phase: out ~ cos(w k c + phase_o); need phase_o ~ ... a+b: cos(k a+pa)cos(k b+pb) -> cos(k(a+b)+pa+pb); out = cos(k c - (pa+pb))?
    err=np.angle(np.exp(1j*(pa[k-1]+pb[k-1]+po[k-1])))
    cnt[k]+=1
    # second freq
    srt=np.argsort(-tot); k2=srt[1]+1
    print(f"{n:4d} k={k:2d} fa={fa:.2f} fb={fb:.2f} fo={fo:.2f} err={err:+.2f} k2={k2:2d} ({tot[k2-1]/3:.2f}) b={b1[n]:+.3f}")
print(sorted(cnt.items()))
