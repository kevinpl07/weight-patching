import numpy as np, json
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)
kill=[3,8,56,68,69,98,77,79,96]
cleanN=[37,65]
nb2=b2.astype(np.float64).copy()
ops=[]
for i in kill: nb2+=W2[:,i]*h[i].mean()
ops.append({"tensor":"fc1.bias","rows":kill,"cols":None,"op":"set","value":-10.0})
for i in cleanN:
    k=dom[i]
    ra=projmean(W1[i,:p],k); rb=projmean(W1[i,p:],k); co=proj(W2[:,i],k)
    hc=np.maximum(ra[A]+rb[B]+b1[i],0)
    nb2+=W2[:,i]*h[i].mean()-co*hc.mean()
    ops.append({"tensor":"fc1.weight","rows":[i],"cols":None,"op":"set","value":[np.concatenate([ra,rb]).tolist()]})
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[i],"op":"set","value":[[float(x)] for x in co]})
ops.append({"tensor":"fc2.bias","rows":None,"cols":None,"op":"set","value":nb2.tolist()})
json.dump(ops,open(S+"scratch/edit.json","w"))
print("ok")
