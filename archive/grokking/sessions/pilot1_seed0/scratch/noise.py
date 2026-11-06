import numpy as np
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
Wa=W1[:,:p]; Wb=W1[:,p:]
al=np.load(S+"scratch/align.npy")
H=np.maximum(Wa[:,A].T+Wb[:,B].T+b1,0)
def comps(lg):
    lg=lg-lg.mean(1,keepdims=True)
    tot=(lg**2).mean()
    # a-only: mean over pairs with same a
    Na=np.zeros((p,p)); Nb=np.zeros((p,p))
    for x in range(p): Na[x]=lg[A==x].mean(0); Nb[x]=lg[B==x].mean(0)
    # sum-structured: logit at c depends on (a+b-c)
    d=(A[:,None]+B[:,None]-np.arange(p)[None,:])%p
    Ps=np.array([lg[d==k].mean() for k in range(p)])
    Ad=(A[:,None]-B[:,None]+0*np.arange(p))  # placeholder
    return tot,(Na[A]**2).mean(),(Nb[B]**2).mean(),(Ps[d]**2).mean(),Ps[0],np.delete(Ps,0).std()
F=al>3
for name,m in [("F",F),("M",~F),("all",np.ones(128,bool))]:
    lg=H[:,m]@W2[:,m].T
    print(name,"total var %.2f a-only %.2f b-only %.2f sum-struct %.2f | P(0)=%.2f other std %.2f"%comps(lg))
