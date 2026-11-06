import numpy as np, json, sys
p=97
d=np.load("train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
def logits(W):
    H=np.maximum(W["fc1.weight"][:,a]+W["fc1.weight"][:,p+b]+W["fc1.bias"][:,None],0).T
    return H@W["fc2.weight"].T+W["fc2.bias"]
def proxy(W):
    L=logits(W); L=L-L.mean(1,keepdims=True)
    tot=L.var(0).sum()
    within=0
    M=np.zeros((p,p))
    for s in range(p):
        m=y==s; M[s]=L[m].mean(0); within+=((L[m]-M[s])**2).sum()
    within/=len(y)
    # "s-mean" classifier: does the group-mean logit predict s? (diag dominance)
    return dict(within_frac=within/tot, groupmean_diag_margin=float(np.mean(np.diag(M)-np.sort(M,1)[:,-2])))
def apply(W,ops):
    W={k:v.copy() for k,v in W.items()}
    for o in ops:
        W[o["tensor"]][o["rows"][0],o["cols"][0]]=o["value"][0][0]
    return W
W0=dict(np.load("weights.npz"))
print("orig",proxy(W0))
for f in sys.argv[1:]:
    print(f,proxy(apply(W0,json.load(open(f)))))
