import numpy as np
D="<project>/runs/seed1/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
Wa,Wb=W1[:,:P],W1[:,P:]
def spec(v): return np.abs(np.fft.rfft(v-v.mean()))**2
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]; return np.fft.irfft(G,P)
alive=[i for i in range(128) if np.linalg.norm(Wa[i])>1]
K={}
for i in alive:
    t=spec(Wa[i])+spec(Wb[i])+spec(W2[:,i]); K[i]=np.argmax(t[1:])+1
c=np.arange(P)
H=np.maximum(Wa[:,a].T+Wb[:,b].T+b1,0)
def analyze(clean):
    groups={}
    for i in alive: groups.setdefault(K[i],[]).append(i)
    tot=np.zeros((len(y),P))
    print("clean" if clean else "full")
    for k,ns in sorted(groups.items()):
        L=np.zeros((len(y),P))
        for i in ns:
            if clean:
                h=np.maximum(proj(Wa[i],k)[a]+proj(Wb[i],k)[b]+b1[i],0); o=proj(W2[:,i],k)
            else:
                h=H[:,i]; o=W2[:,i]
            L+=np.outer(h,o)
        L-=L.mean(1,keepdims=True)
        ph=2*np.pi*k*((a+b)[:,None]-c[None,:])/P
        Cc,Ss=np.cos(ph),np.sin(ph)
        ac=(L*Cc).sum(1)/(Cc**2).sum(1); as_=(L*Ss).sum(1)/(Ss**2).sum(1)
        res=L-ac[:,None]*Cc-as_[:,None]*Ss
        print("k=%2d n=%d sig=%.2f (sin %.2f)  sig_std=%.2f  resid_rms=%.2f"%(k,len(ns),ac.mean(),as_.mean(),ac.std(),np.sqrt((res**2).mean())))
        tot+=L
    return tot
for cl in [False,True]:
    L=analyze(cl)
    print("acc", (L.argmax(1)==y).mean())
