import numpy as np, sys, json
S=sys.argv[1]; th=float(sys.argv[2]); out=sys.argv[3]
W=dict(np.load(S+"/weights.npz")); p=97
W1=W["fc1.weight"]; frac=np.load(S+"/scratch/frac.npy")
kill=np.where(frac<th)[0]
mx=W1[:,:p].max(1)+W1[:,p:].max(1)
vals=[float(-(mx[n])-1.0) for n in kill]
ops=[{"tensor":"fc1.bias","rows":kill.tolist(),"cols":None,"op":"set","value":vals},
     {"tensor":"fc1.bias","rows":kill.tolist(),"cols":None,"op":"reset_optimizer"}]
json.dump(ops,open(out,"w"))
print(len(kill), kill.tolist(), np.round(vals,2))
