from lib import *
print('real', metr(fwd(w1,b1,w2,b2)))
W1,B1,W2=ideal(); print('ideal k-only', metr(fwd(W1,B1,W2)))
W1,B1,W2=ideal(out_ks=(1,2,3)); print('ideal out k,2k,3k', metr(fwd(W1,B1,W2)))
# scale matters for loss; report acc mostly
from collections import Counter
cnt=Counter(K.values())
single=[i for i in live if cnt[K[i]]==1]
W1,B1,W2=ideal([i for i in live if i not in single]); print('ideal no singletons', metr(fwd(W1,B1,W2)))
print('real no singletons', metr(fwd(w1,b1,np.where(np.isin(np.arange(128),single)[None,:],0,w2))))
# per-frequency group quality on train: signal = logit[y]-mean, noise = std over c
W1,B1,W2=ideal()
for k in sorted(cnt):
    ns=[i for i in live if K[i]==k]
    H=np.maximum(W1[ns][:,A].T+W1[ns][:,p+B].T+B1[ns],0); L=H@W2[:,ns].T
    L=L-L.mean(0)  # remove fixed bias pattern
    sig=L[np.arange(len(Y)),Y]-L.mean(1)
    print(k,len(ns),'signal %.2f noise std %.2f'%(sig.mean(), L.std(1).mean()), 'sig std %.2f'%sig.std())
