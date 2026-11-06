import numpy as np, json
exec(open("scratch/newneur.py").read().split('report("base"')[0].replace("t=-0.3","t=-0.25"))
X1,xb,X2,Q1,Q2,slots=build([7,13],"quad")
hf=H(X1,xb).mean(1); b2dc=-X2@hf; b2dc+=b2.mean()-b2dc.mean()
delta=np.abs(b2dc-b2); keep=np.argsort(delta)[:5]
b2n=b2dc.copy(); b2n[keep]=b2[keep]
print("slots",slots,"skipped b2",keep,delta[keep])
print("final full",ev(X1,xb,X2,b2n))
bp=xb.copy(); bp[[i for i in noisy if i not in slots]]=-4
print("final clean proj",ev(Q1,bp,Q2,-Q2@H(Q1,bp).mean(1)))
ops=[]
ops.append({"tensor":"fc1.bias","rows":[int(i) for i in range(128)],"op":"set","value":[float(v) for v in xb]})
for s in slots:
    ops.append({"tensor":"fc1.weight","rows":[int(s)],"cols":None,"op":"set","value":[[float(v) for v in X1[s]]]})
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[int(s)],"op":"set","value":[[float(v)] for v in X2[:,s]]})
ops.append({"tensor":"fc2.bias","rows":list(range(97)),"op":"set","value":[float(v) for v in b2n]})
json.dump(ops,open("scratch/edit.json","w"))
nchg=int((xb!=b1).sum()+(X1!=W1).sum()+(X2!=W2).sum()+(b2n!=b2).sum()); print("changed",nchg)
