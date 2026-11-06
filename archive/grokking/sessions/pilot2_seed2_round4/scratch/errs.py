import numpy as np, collections
D="<project>/runs/seed2/loop/round4/agent/"
W=dict(np.load(D+"weights.npz")); p=97
W1,b1,W2,b2=W['fc1.weight'],W['fc1.bias'],W['fc2.weight'],W['fc2.bias']
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
A=W1[:,:p];B=W1[:,p:];O=W2.T
def spec(X):
    F=np.fft.rfft(X,axis=1);P=np.abs(F)**2;P[:,0]=0;return F,P
FA,PA=spec(A);FB,PB=spec(B);FO,PO=spec(O)
f=(PA+PB+PO).argmax(1)
conc=np.array([(PA[n,f[n]]/PA[n].sum()+PB[n,f[n]]/PB[n].sum()+PO[n,f[n]]/PO[n].sum())/3 for n in range(128)])
mask=(conc>0.6).astype(float)
h=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)*mask
c2=-(W2@h.mean(0)); c2-=c2.mean()
lg=h@W2.T+c2
pred=lg.argmax(1); wrong=np.where(pred!=y)[0]
print("n wrong",len(wrong))
d=(pred[wrong]-y[wrong])%p
print("pred-y offsets:",sorted(collections.Counter(np.minimum(d,p-d)).items(),key=lambda t:-t[1])[:15])
# profile g(d): mean logit at c=y+d minus mean
G=np.zeros(p)
for dd in range(p): G[dd]=(lg[np.arange(len(y)),(y+dd)%p]).mean()
G-=G.mean(); print("profile top:",np.argsort(-G)[:8], np.sort(G)[::-1][:8].round(2))
print("errors by y:",sorted(collections.Counter(y[wrong]).items(),key=lambda t:-t[1])[:10])
print("errors by a:",sorted(collections.Counter(a[wrong]).items(),key=lambda t:-t[1])[:10])
print("errors by b:",sorted(collections.Counter(b[wrong]).items(),key=lambda t:-t[1])[:10])
print("c2",c2.round(2)[:20], "old b2", b2[:20].round(2))
