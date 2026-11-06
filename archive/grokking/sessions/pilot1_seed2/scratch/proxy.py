import numpy as np
D=r"<project>\runs\seed2\agent"
p=97
T=np.load(D+r"\train_pairs.npy"); A,B,Y=T[:,0],T[:,1],T[:,2]
def logits(W):
    h=np.maximum(W["fc1.weight"][:,A].T+W["fc1.weight"][:,p+B].T+W["fc1.bias"],0)
    return h@W["fc2.weight"].T+W["fc2.bias"]
def proxy(W,seed=0):
    L=logits(W); acc=(L.argmax(1)==Y).mean()
    idx=(Y[:,None]+np.arange(p)[None,:])%p
    R=np.take_along_axis(L,idx,1); R=R-R.mean(1,keepdims=True)
    rb=R.mean(0); dev=R-rb
    rng=np.random.default_rng(seed); js=rng.integers(1,p,len(Y))
    Z=rb[None,:]+dev
    z0=rb[0]+dev[np.arange(len(Y)),js]
    Z2=Z.copy(); Z2[:,0]=-1e9; Z2[np.arange(len(Y)),js]=-1e9
    est=(z0>Z2.max(1)).mean()
    margin=rb[0]-np.sort(rb[1:])[-1]
    return dict(train_acc=round(acc,4),proxy_heldout=round(est,4),signal_margin=round(margin,3),
                noise_std_off=round(dev[:,1:].std(),3),dev0_mean=round(dev[:,0].mean(),3),rb0=round(rb[0],3))
if __name__=="__main__":
    W=dict(np.load(D+r"\weights.npz")); print("orig",proxy(W))
