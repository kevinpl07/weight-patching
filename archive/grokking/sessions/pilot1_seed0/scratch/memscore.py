import numpy as np
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
Wa=W1[:,:p]; Wb=W1[:,p:]
al=np.load(S+"scratch/align.npy")
H=np.maximum(Wa[:,A].T+Wb[:,B].T+b1,0)   # N x 128
n=len(Y)
# per neuron: empirical additive components on train
memA=np.zeros(128); memB=np.zeros(128); inter=np.zeros(128)
for j in range(128):
    lg=np.outer(H[:,j],W2[:,j]); lg-=lg.mean(1,keepdims=True)
    Na=np.zeros((p,p)); Nb=np.zeros((p,p))
    for x in range(p): Na[x]=lg[A==x].mean(0); Nb[x]=lg[B==x].mean(0)
    memA[j]=Na[A,Y].mean(); memB[j]=Nb[B,Y].mean()   # boost of train labels via a-only part
    inter[j]=(lg[np.arange(n),Y]-Na[A,Y]-Nb[B,Y]).mean()
o=np.argsort(al)
for j in o: print("%3d al %.2f  addmem %.3f  inter %.3f  bias %.2f"%(j,al[j],memA[j]+memB[j],inter[j],b1[j]))
print("sum addmem F",(memA+memB)[al>3].sum(),"M",(memA+memB)[al<=3].sum())
print("sum inter F",inter[al>3].sum(),"M",inter[al<=3].sum())
np.save(S+"scratch/addmem.npy",memA+memB)
