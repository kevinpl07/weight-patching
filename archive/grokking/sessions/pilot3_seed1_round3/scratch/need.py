import numpy as np, json
S="<project>/runs/seed1/loop2/round3/agent/"
p=97; d=np.load(S+"train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
for f in ("weights.npz","working.npz"):
    W=dict(np.load(S+f)); W1=W['fc1.weight']; b1=W['fc1.bias']; W2=W['fc2.weight']; b2=W['fc2.bias']
    h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2
    cm=L.mean(0); print(f,"classmean (incl bias) sorted", np.round(np.sort(cm-cm.mean()),2)[[0,1,2,3,-4,-3,-2,-1]], "argmax",np.argmax(np.abs(cm-cm.mean())), "std",cm.std())
    # mean logit per class restricted to pairs where class is NOT correct
    M=np.ones_like(L,bool); M[np.arange(len(y)),y]=False
    cmw=(L*M).sum(0)/M.sum(0); print("  wrong-only classmean std",cmw.std(), "counts of argmax class", np.bincount(y)[np.argmax(np.abs(cm-cm.mean()))])
ops=json.load(open(S+"scratch/editB.json")); from collections import Counter; print(Counter(o["tensor"] for o in ops))
