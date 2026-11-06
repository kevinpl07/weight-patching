import numpy as np, json, sys
thr=float(sys.argv[1])
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; b1=W["fc1.bias"]
c=np.load("scratch/conc.npy")
kill=np.where(c<=thr)[0]
mx=w1[:,:p].max(1)+w1[:,p:].max(1)+b1
print("n kill",len(kill),"max possible pre among killed",mx[kill].max())
# bias chosen so neuron is dead for every (a,b): -(max_a A + max_b B) - 1
vals=[float(-(w1[i,:p].max()+w1[i,p:].max())-1.0) for i in kill]
ops=[{"tensor":"fc1.bias","rows":kill.tolist(),"cols":None,"op":"set","value":vals}]
json.dump(ops,open(f"scratch/kill_{thr}.json","w"))
print(np.round(vals,2))
