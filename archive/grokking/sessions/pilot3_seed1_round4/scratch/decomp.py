# decompose train logits of the (possibly edited) network: offset(c), f(a,c), g(b,c), rest
import numpy as np,sys
D="<project>/runs/seed1/loop2/round4/agent/"
W=dict(np.load(D+sys.argv[1])); p=97
T=np.load(D+"train_pairs.npy"); a,b,y=T[:,0],T[:,1],T[:,2]
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0); L=h@W2.T+b2
acc=lambda L:(L.argmax(1)==y).mean()
Lc=L-L.mean(0)
fa=np.zeros((p,p)); gb=np.zeros((p,p))
for j in range(p): fa[j]=Lc[a==j].mean(0)
for j in range(p): gb[j]=Lc[b==j].mean(0)
R=Lc-fa[a]-gb[b]
print("std: offset %.3f f(a,c) %.3f g(b,c) %.3f rest %.3f"%(L.mean(0).std(),fa.std(),gb.std(),R.std()))
print("acc full %.4f | no offset %.4f | no offset,f,g %.4f"%(acc(L),acc(Lc),acc(R)))
for name,M in [("f",fa),("g",gb)]:
    M=M-M.mean(0)
    U,S,Vt=np.linalg.svd(M); e=S**2/np.sum(S**2)
    print(name,"top sv",np.round(S[:8],2),"cum energy",np.round(np.cumsum(e)[:8],3))
    print("  max|u1|*S1*max|v1|=%.3f"%(S[0]*np.abs(U[:,0]).max()*np.abs(Vt[0]).max()))
Uf,Sf,Vf=np.linalg.svd(fa-fa.mean(0)); Ug,Sg,Vg=np.linalg.svd(gb-gb.mean(0))
print("overlap of top c-directions f vs g:",np.round(np.abs(Vf[:6]@Vg[:6].T),2))
