import numpy as np, json
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
T=np.load(D+"train_pairs.npy"); A,B=T[:,0],T[:,1]
p=97
h=np.maximum(W1[:,A].T+W1[:,p+B].T+b1,0)
mem=[0,9,10,12,13,15,17]
off=W2[:,mem]@h[:,mem].mean(0)
new=[22,24,25]; H=2.0
o=(off/(H*len(new))).astype(float)
ops=[{"tensor":"fc1.bias","rows":mem,"cols":None,"op":"set","value":-10.0}]
for i in new:
    ops.append({"tensor":"fc1.weight","rows":[i],"cols":list(range(p)),"op":"set","value":[[1.0]*p]})
    ops.append({"tensor":"fc1.bias","rows":[i],"cols":None,"op":"set","value":1.0})
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[i],"op":"set","value":[[float(v)] for v in o]})
json.dump(ops,open(D+"scratch/edit.json","w"))
print("max |o|",np.abs(o).max())
