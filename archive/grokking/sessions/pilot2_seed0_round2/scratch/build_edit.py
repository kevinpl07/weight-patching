import numpy as np, json, sys
exec(open("<project>/runs/seed0/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
FOLD_SPIKES = len(sys.argv)>1 and sys.argv[1]=="fold"
TARGET3 = not (len(sys.argv)>2 and sys.argv[2]=="konly")
if not TARGET3: OUTK=lambda k:[k]
C=clean(W)
V={k:v.copy() for k,v in W.items()}
# 1) mean-ablate memorizing neurons: kill via fc1 bias, fold training-mean activation into fc2 bias
for i in mem:
    V["fc2.bias"]+=mact[i]*w2[:,i]
    V["fc1.bias"][i]=-(w1[i,:p].max()+w1[i,p:].max()+1.0)
# 2) set the 637 largest-|residual| fc2 entries of Fourier neurons to their clean (k[,3k]) projection
R2=np.zeros_like(w2); R2[:,four]=w2[:,four]-C["fc2.weight"][:,four]
K=749-2*len(mem)-p+len(mem)  # 15 fc1.bias + 97 fc2.bias
K=749-len(mem)-p
thr=np.sort(np.abs(R2).ravel())[-K]; m=np.abs(R2)>=thr
V["fc2.weight"][m]=C["fc2.weight"][m]
if FOLD_SPIKES: V["fc2.bias"]+=((w2-V["fc2.weight"])*mact[None,:]).sum(1)
print("K",K,"spike entries",m.sum(),"neurons touched",np.unique(np.where(m)[1]).size)
print("result",fwd(V))
ch={k:int((V[k]!=W[k]).sum()) for k in W}; print(ch,sum(ch.values()))
ops=[{"tensor":"fc1.bias","rows":[int(i) for i in mem],"cols":None,"op":"set","value":[float(V["fc1.bias"][i]) for i in mem]},
     {"tensor":"fc2.bias","rows":None,"cols":None,"op":"set","value":[float(x) for x in V["fc2.bias"]]}]
for r,c in zip(*np.where(m)):
    ops.append({"tensor":"fc2.weight","rows":[int(r)],"cols":[int(c)],"op":"set","value":[[float(V["fc2.weight"][r,c])]]})
json.dump(ops,open("<project>/runs/seed0/loop/round2/agent/scratch/edit.json","w"))
np.savez("<project>/runs/seed0/loop/round2/agent/scratch/expected.npz",**V)
