import numpy as np, io, contextlib
exec(open('proxy.py').read().split('singles=')[0].replace('print("orig"','#'))
src=open('decomp.py').read().split('L=h.T@W2.T')[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src.replace('W=dict(np.load(D+"weights.npz"))','W=W'))
def logits(V):
    h=np.maximum(V["fc1.weight"][:,A_]+V["fc1.weight"][:,p+B_]+V["fc1.bias"][:,None],0)
    return h.T@V["fc2.weight"].T
for nm,V in [("orig",W),("proj",P)]:
    L=logits(V); print(nm,{k:round(float(v),3) for k,v in decomp(L).items()})
# harmonic-richer projections
for hs in [[1,2,3,4,5],[1,2,3,4,5,6,7,8]]:
    Q={k:v.copy() for k,v in W.items()}
    for n in live:
        ks=[(h*kof(n))%p for h in hs]
        Q["fc2.weight"][:,n]=proj(W["fc2.weight"][:,n],ks)
        Q["fc1.weight"][n,:p]=proj(W1[n,:p],ks); Q["fc1.weight"][n,p:]=proj(W1[n,p:],ks)
    print(hs,fwd(Q))
