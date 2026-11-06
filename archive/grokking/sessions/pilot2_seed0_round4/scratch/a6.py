from lib import *
tot1=tot2=0
rows=[]
for i in live:
    k=K[i]; ks=[k,2*k%p,3*k%p]
    ra=w1[i,:p]-proj(w1[i,:p],ks); rb=w1[i,p:]-proj(w1[i,p:],ks); ro=w2[:,i]-proj(w2[:,i],ks,const=True)
    r1=np.concatenate([ra,rb])
    rows.append((i,k,np.linalg.norm(r1),np.abs(r1).max(),np.linalg.norm(ro),np.abs(ro).max()))
    tot1+=np.sum(r1**2); tot2+=np.sum(ro**2)
for r in rows: print("n%3d k%2d fc1res %.2f max %.2f  fc2res %.2f max %.2f"%r)
print('tot',tot1**.5,tot2**.5, 'fc1 norm', np.linalg.norm(w1), 'fc2', np.linalg.norm(w2))
# ideal with harmonics
def ideal2(ks_fn):
    W1=np.zeros_like(w1); W2=np.zeros_like(w2)
    for i in live:
        ks=ks_fn(K[i])
        W1[i,:p]=proj(w1[i,:p],ks); W1[i,p:]=proj(w1[i,p:],ks); W2[:,i]=proj(w2[:,i],ks,const=True)
    return W1,b1.copy(),W2
for name,fn in [('k',lambda k:[k]),('k2k',lambda k:[k,2*k%p]),('k2k3k',lambda k:[k,2*k%p,3*k%p])]:
    W1,B1,W2=ideal2(fn); print(name, metr(fwd(W1,B1,W2)), metr(fwd(W1,B1,W2,b2)))
