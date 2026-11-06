"""Build seed edit: twins for singleton neurons, rotated twin pairs (+pi/2 on a and b) for twin groups.
usage: build.py OUT.json NA NB "twin:91,96" "rot:10,13,..." """
import numpy as np, json, sys
S="<project>/runs/seed2/loop2/round4/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,w2=W["fc1.weight"],W["fc2.weight"]
A=w1[:,:p];B=w1[:,p:]
out,NA,NB=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
twins=[int(x) for x in sys.argv[4].split(":")[1].split(",") if x]
rots=[int(x) for x in sys.argv[5].split(":")[1].split(",") if x]
scale=float(sys.argv[6]) if len(sys.argv)>6 else 1.0
dead=[i for i in range(128) if np.linalg.norm(w1[i])<1e-3]
groups={}
for i in range(128):
    if np.linalg.norm(w1[i])<1 or i==30: continue
    F=np.abs(np.fft.rfft(A[i]))**2;F[0]=0;groups.setdefault(int(np.argmax(F)),[]).append(i)
x=np.arange(p)
def proj(v,k,shift):
    F=np.fft.rfft(v)[k]; return 2*abs(F)/p*np.cos(2*np.pi*k*x/p+np.angle(F)+shift)
def sparse(v,n):
    o=np.zeros(p); idx=np.argsort(-np.abs(v))[:n]; o[idx]=v[idx]; return o
seeds=[]  # (row a, row b, desc)
for n in twins:
    F=np.abs(np.fft.rfft(A[n]))**2;F[0]=0;k=int(np.argmax(F))
    seeds.append((-proj(A[n],k,0),-proj(B[n],k,0),f"twin of {n} k{k}"))
for k in rots:
    n=groups[k][0]
    a,b=proj(A[n],k,np.pi/2),proj(B[n],k,np.pi/2)
    seeds.append((a,b,f"rot+ k{k} ref {n}")); seeds.append((-a,-b,f"rot- k{k} ref {n}"))
ops=[];used=0
for (a,b,d),r in zip(seeds,dead):
    row=np.concatenate([sparse(a,NA),sparse(b,NB)])*scale
    cols=[int(c) for c in np.nonzero(row)[0]]
    ops.append({"tensor":"fc1.weight","rows":[r],"cols":cols,"op":"set","value":[[float(row[c]) for c in cols]]})
    ops.append({"tensor":"fc1.weight","rows":[r],"cols":cols,"op":"reset_optimizer"})
    used+=len(cols); print(r,d,len(cols))
print("total",used)
json.dump(ops,open(out,"w"))
