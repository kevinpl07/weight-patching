import numpy as np
S="<project>/runs/seed0/loop2/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
Wa=W1[:,:p]; Wb=W1[:,p:]
live=np.where(np.linalg.norm(W1,axis=1)>1)[0]
print("live",len(live))
def spec(v):
    F=np.fft.rfft(v); P=np.abs(F)**2; P[1:]*=2; return F,P
rows=[]
for n in live:
    Fa,Pa=spec(Wa[n]); Fb,Pb=spec(Wb[n]); Fo,Po=spec(W2[:,n])
    tot=Pa+Pb+Po
    k=np.argmax(tot[1:])+1
    ca=Pa[k]/Pa.sum(); cb=Pb[k]/Pb.sum(); co=Po[k]/Po.sum()
    pa=np.angle(Fa[k]); pb=np.angle(Fb[k]); po=np.angle(Fo[k])
    # phase relation: a,b phases ~ alpha, beta; output phase should be ~ alpha+beta
    rows.append((n,k,ca,cb,co,pa,pb,po,(pa+pb-po+np.pi)%(2*np.pi)-np.pi, Pa[0]/Pa.sum(),Po[0]/Po.sum()))
rows.sort(key=lambda r:r[1])
print(" n   k  conc_a conc_b conc_o  ph_a  ph_b ph_o  (a+b-o) DCa DCo")
for r in rows: print("%3d %3d  %.2f %.2f %.2f  %+.2f %+.2f %+.2f  %+.2f  %.3f %.3f"%r)
from collections import Counter
print(Counter(r[1] for r in rows))
# fc2 bias spectrum
F,P=spec(b2); print("b2 power top", sorted([(round(P[k]/P.sum(),3),k) for k in range(49)])[-8:])
print("b2", np.round(b2[:20],2))
