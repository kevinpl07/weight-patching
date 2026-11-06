import numpy as np, json
p=97
d=np.load("train_pairs.npy"); a,b,y=d[:,0],d[:,1],d[:,2]
W0=dict(np.load("weights.npz")); ops=json.load(open("scratch/edit_flip.json"))
def apply(W,ops):
    W={k:v.copy() for k,v in W.items()}
    for o in ops: W[o["tensor"]][o["rows"][0],o["cols"][0]]=o["value"][0][0]
    return W
def ev(W):
    H=np.maximum(W["fc1.weight"][:,a]+W["fc1.weight"][:,p+b]+W["fc1.bias"][:,None],0).T
    L=H@W["fc2.weight"].T+W["fc2.bias"]; L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    Lc=L-L.mean(1,keepdims=True); within=sum(((Lc[y==s]-Lc[y==s].mean(0))**2).sum() for s in range(p))/len(y)
    return round((L.argmax(1)==y).mean(),4), round(float(-lp[np.arange(len(y)),y].mean()),4), round(within/Lc.var(0).sum(),4)
W1=apply(W0,ops)
chosen={o["rows"][0] for o in ops if o["tensor"]=="fc1.weight"}|{o["cols"][0] for o in ops if o["tensor"]=="fc2.weight"}
noisy=[i for i in range(128) if i not in chosen]
print("n noisy",len(noisy))
print("flip",ev(W1))
for bv in [0.0,-0.3,-0.6,-1.0,-2.0]:
    W=apply(W0,ops); W["fc1.bias"][noisy]=bv; print("bias",bv,ev(W))
