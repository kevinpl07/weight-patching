import numpy as np, json
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64)
tr=np.load(S+"train_pairs.npy")
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),tr[:,0]]=1; X[np.arange(len(tr)),p+tr[:,1]]=1
H=np.maximum(X@W1.T+b1,0)
act=(H>0).mean(0)
alive=[int(n) for n in range(128) if act[n]>0.05]
dead=[int(n) for n in range(128) if act[n]==0]
print(len(alive),len(dead))
pre=X@W1[dead].T
print("revived active frac with bias0:",np.round(((pre>0).mean(0)),2))
ops=[{"tensor":"fc1.bias","rows":alive,"cols":None,"op":"add","value":-0.25},
     {"tensor":"fc1.bias","rows":dead,"cols":None,"op":"set","value":0.0}]
json.dump(ops,open(S+"scratch/edit.json","w"))
