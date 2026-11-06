import numpy as np
S="<project>/runs/seed0/loop2/round2/agent/"
W0=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
H=np.maximum(0,W0['fc1.weight'][:,a].T+W0['fc1.weight'][:,p+b].T+W0['fc1.bias'])
Z=H@W0['fc2.weight'].T
def ev(b2):
    L=Z+b2; L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return round(float(-lp[np.arange(len(y)),y].mean()),4), round(float((L.argmax(1)==y).mean()),4)
off=H.mean(0)@W0['fc2.weight'].T
cnt=np.bincount(y,minlength=p); lpr=np.log(cnt/cnt.sum()); lpr-=lpr.mean()
print("cur",ev(W0['fc2.bias'])); print("-off",ev(-off)); print("-off+lprior",ev(-off+lpr)); print("clip(-off)",ev(np.clip(-off,-1,1)))
print("b2 entries at |.|>0.95:",(np.abs(W0['fc2.bias'])>0.95).sum())
o=dict(np.load(S+"optimizer.npz")); 
