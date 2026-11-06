import numpy as np
S="<project>/runs/seed1/loop/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
D=np.load(S+"train_pairs.npy"); A,B,Y=D[:,0],D[:,1],D[:,2]
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
alive=[i for i in range(128) if np.linalg.norm(W1[i])>1]
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]*0; return np.fft.irfft(G,n=len(v))
def projmean(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]; return np.fft.irfft(G,n=len(v))
def fwd(W1,b1,W2,b2):
    h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)  # 128 x N
    L=(W2@h).T+b2
    L=L-L.max(1,keepdims=True); lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    return (L.argmax(1)==Y).mean(), -lp[np.arange(len(Y)),Y].mean()
print("orig",fwd(W1,b1,W2,b2))
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
def clean(which, keepmean=True, kill=()):
    V1=W1.copy(); V2=W2.copy(); c1=b1.copy()
    pr=projmean if keepmean else proj
    for i in alive:
        k=dom[i]
        if 'a' in which: V1[i,:p]=pr(W1[i,:p],k)
        if 'b' in which: V1[i,p:]=pr(W1[i,p:],k)
        if 'o' in which: V2[:,i]=pr(W2[:,i],k)
    for i in kill: c1[i]=-10
    return V1,c1,V2,b2
for w in ["ab","o","abo"]:
    for km in [True,False]:
        print(w,km,fwd(*clean(w,km)))
bad=[3,8,56,68,69,98]
bad2=bad+[96,79,77,37,65,33]
print("kill bad",fwd(*clean("",kill=bad)))
print("kill bad2",fwd(*clean("",kill=bad2)))
print("clean+kill bad",fwd(*clean("abo",False,kill=bad)))
print("clean+kill bad2",fwd(*clean("abo",False,kill=bad2)))
# mean (DC) component sizes
for i in alive[:5]: print(i,W1[i,:p].mean(),W1[i,p:].mean(),W2[:,i].mean())
