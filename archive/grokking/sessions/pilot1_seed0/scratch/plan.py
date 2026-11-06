import numpy as np, json, sys
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
al=np.load(S+"scratch/align.npy")
F=np.where(al>3)[0]
def metrics(W1n,b1n,W2n,b2n):
    H=np.maximum(W1n[:,A].T+W1n[:,p+B].T+b1n,0); lg=H@W2n.T+b2n
    lg=lg-lg.max(1,keepdims=True); lp=lg-np.log(np.exp(lg).sum(1,keepdims=True))
    return round((lg.argmax(1)==Y).mean(),3), round(float(-lp[np.arange(len(Y)),Y].mean()),3)
# partner search: for each F neuron (by alignment), find non-F neuron minimizing mismatches
def mism(i,j):
    r=np.sign(W1[j])!=np.sign(-W1[i]); c=np.sign(W2[:,j])!=np.sign(W2[:,i])
    return r.sum()+c.sum(), r, c
NPART=int(sys.argv[1]); TH=float(sys.argv[2])
cand=[j for j in range(128) if al[j]<1.0]  # pool for partners: low alignment memorizers
used=set(); partners=[]
for i in F[np.argsort(-al[F])][:NPART]:
    best=min((mism(i,j)[0],j) for j in cand if j not in used)
    used.add(best[1]); partners.append((int(i),best[1],int(best[0])))
print("partners (F, j, changes):",partners)
W1n,b1n,W2n=W1.copy(),b1.copy(),W2.copy()
ops=[]
for i,j,_ in partners:
    _,r,c=mism(i,j)
    cols=np.where(r)[0]; W1n[j,cols]=-W1[i,cols]
    ops.append({"tensor":"fc1.weight","rows":[j],"cols":cols.tolist(),"op":"set","value":[(-W1[i,cols]).tolist()]})
    rows=np.where(c)[0]; W2n[rows,j]=W2[rows,i]
    ops.append({"tensor":"fc2.weight","rows":rows.tolist(),"cols":[j],"op":"set","value":[[float(v)] for v in W2[rows,i]]})
    b1n[j]=b1[i]; ops.append({"tensor":"fc1.bias","rows":[j],"op":"set","value":[float(b1[i])]})
kill=[j for j in range(128) if al[j]<TH and j not in used]
b1n[kill]=-3.0
ops.append({"tensor":"fc1.bias","rows":kill,"op":"set","value":[-3.0]*len(kill)})
n=sum((W1n!=W1).sum() for _ in [0])+(b1n!=b1).sum()+(W2n!=W2).sum()
print("kills",len(kill),"alive",128-len(kill),"entries",int(n))
print("train orig",metrics(W1,b1,W2,b2),"new",metrics(W1n,b1n,W2n,b2))
# partner quality
for i,j,_ in partners: print(i,j,"corr row",np.corrcoef(W1n[j],-W1[i])[0,1].round(3),"corr col",np.corrcoef(W2n[:,j],W2[:,i])[0,1].round(3))
json.dump(ops,open(S+"scratch/edit_%d_%s.json"%(NPART,TH),"w"))
