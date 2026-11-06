from lib import *
W1,B1,W2=ideal()
L=fwd(W1,B1,W2)
ok=(L.argmax(1)==Y)
Ls=L-L.max(1,keepdims=True); lp=Ls-np.log(np.exp(Ls).sum(1,keepdims=True)); nll=-lp[np.arange(len(Y)),Y]
D=(A-B)%p
acc=np.array([ok[D==t].mean() for t in range(p)]); cnt=np.array([(D==t).sum() for t in range(p)])
nl=np.array([nll[D==t].mean() for t in range(p)])
o=np.argsort(acc)
print('per diag acc (worst 20):'); print([(int(t),round(acc[t],2),int(cnt[t])) for t in o[:20]])
print('acc quantiles',np.percentile(acc,[0,10,25,50,75,100]))
S=(A+B)%p
acc2=np.array([ok[S==t].mean() for t in range(p)]); print('by sum: min',acc2.min(),'std',acc2.std(),'diag std',acc.std())
# real model margin by diag
Lr=fwd(w1,b1,w2,b2); m=Lr[np.arange(len(Y)),Y]-np.where(np.eye(p,dtype=bool)[Y],-1e9,Lr).max(1)
print('real margin by worst diag', [round(m[D==t].mean(),2) for t in o[:10]], 'overall', m.mean())
