import numpy as np, json
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
T=np.load(D+"train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
p=97
h=np.maximum(W1[:,A].T+W1[:,p+B].T+b1,0)
mem=[0,9,10,12,13,15,17]
off=W2[:,mem]@h[:,mem].mean(0)   # mean logit contribution of memorizers
print("mem offset: max|.| %.2f std %.2f mean %.2f"%(np.abs(off).max(),off.std(),off.mean()))
print("sorted", np.round(np.sort(off),2))
new=[22,24,25]
print("new neuron rows norms", [np.abs(W1[i]).sum()+np.abs(W2[:,i]).sum()+abs(b1[i]) for i in new])
