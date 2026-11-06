from lib import *
x=np.arange(p)
print('dead biases', np.round(np.sort(b1[dead])[:5],4), np.round(np.sort(b1[dead])[-5:],4))
def partner(i):
    k=K[i]
    ra=w1[i,:p]; rb=w1[i,p:]
    pa_=proj(ra,[k],const=False); pb_=proj(rb,[k],const=False)
    row=np.concatenate([ra.mean()-pa_, rb.mean()-pb_])
    col=proj(w2[:,i],[k],const=False)
    return row,b1[i],col
def apply(W,adds):
    W={k:v.copy() for k,v in W.items()}
    for j,i in adds:
        r,bb,c=partner(i); W['fc1.weight'][j]=r; W['fc1.bias'][j]=bb; W['fc2.weight'][:,j]=c
    return W
def ev(W,fold=False):
    H=np.maximum(W['fc1.weight'][:,A].T+W['fc1.weight'][:,p+B].T+W['fc1.bias'],0)
    L=H@W['fc2.weight'].T+W['fc2.bias']
    return metr(L)
print('real',ev(W))
sing={15:18,24:54,35:22,21:41,2:34,47:78,38:115,1:39}
for k,i in sing.items():
    print(k, ev(apply(W,[(dead[0],i)])))
print('15+24', ev(apply(W,[(dead[0],18),(dead[1],54)])))
