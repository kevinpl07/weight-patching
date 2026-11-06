import numpy as np
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64); W2=W["fc2.weight"].astype(np.float64)
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
H=np.maximum(X@W1.T+b1,0)
def kof(v):
    F=np.fft.rfft(v-v.mean()); return np.argmax(np.abs(F[1:])**2)+1
alive=[n for n in range(128) if (H[:,n]>0).mean()>0.05]
grp={}
for n in alive: grp.setdefault(kof(W2[:,n]),[]).append(n)
out=[]
for k,ns in sorted(grp.items()):
    D=0;Sg=0
    for n in ns:
        Fa=np.fft.rfft(W1[n,:p])[k]; Fb=np.fft.rfft(W1[n,p:])[k]; Fo=np.fft.rfft(W2[:,n])[k]
        w=abs(Fa)*abs(Fb)*abs(Fo)/1e5
        D+=w*np.exp(1j*(np.angle(Fa)-np.angle(Fb)))
        Sg+=w*np.exp(2j*np.angle(Fo))
    out.append((k,len(ns),abs(D),abs(Sg)))
    print(f"k={k:2d} n={len(ns)} |a-b phasor|={abs(D):.2f} |2phi phasor|={abs(Sg):.2f}")
