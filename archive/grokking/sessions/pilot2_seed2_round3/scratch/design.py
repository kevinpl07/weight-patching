import numpy as np, json
S="<project>/runs/seed2/loop/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W["fc1.weight"].astype(np.float64); b1=W["fc1.bias"].astype(np.float64); W2=W["fc2.weight"].astype(np.float64); b2=W["fc2.bias"].astype(np.float64)
Q=np.load(S+"scratch/proj.npz"); P1,P2=Q["P1"],Q["P2"]
tr=np.load(S+"train_pairs.npy"); a,b,y=tr[:,0],tr[:,1],tr[:,2]
X=np.zeros((len(tr),2*p)); X[np.arange(len(tr)),a]=1; X[np.arange(len(tr)),p+b]=1
j=np.arange(p)
def stats(L):
    corr=L[np.arange(len(y)),y]; o=L.copy(); o[np.arange(len(y)),y]=-1e9
    m=corr-o.max(1); lp=L-L.max(1,keepdims=True); lp=lp-np.log(np.exp(lp).sum(1,keepdims=True))
    return np.round([(m>0).mean(), -lp[np.arange(len(y)),y].mean(), np.median(m)],4)
H=np.maximum(X@W1.T+b1,0)
def kof(v):
    F=np.fft.rfft(v-v.mean()); return np.argmax(np.abs(F[1:])**2)+1
alive=[n for n in range(128) if (H[:,n]>0).mean()>0.05]
grp={}
for n in alive: grp.setdefault(kof(W2[:,n]),[]).append(n)
def design(k,scale=1.0):
    ns=grp[k]; D=0;Sg=0;Aa=[];Ab=[];Ao=[];bs=[]
    for n in ns:
        Fa=np.fft.rfft(W1[n,:p])[k]; Fb=np.fft.rfft(W1[n,p:])[k]; Fo=np.fft.rfft(W2[:,n])[k]
        w=abs(Fa)*abs(Fb)*abs(Fo)
        D+=w*np.exp(1j*(np.angle(Fa)-np.angle(Fb))); Sg+=w*np.exp(2j*np.angle(Fo))
        Aa.append(abs(Fa));Ab.append(abs(Fb));Ao.append(abs(Fo)); bs.append(b1[n])
    dlt=np.angle(D)+np.pi; phi=(np.angle(Sg)+np.pi)/2   # phi_o; also phi+pi possible: choose sign via a/b
    pa=(phi+dlt)/2; pb=(phi-dlt)/2   # pa+pb=phi, pa-pb=dlt
    A=np.mean(Aa+Ab)*scale; Ao_=np.mean(Ao)*scale
    row=np.concatenate([2*A/p*np.cos(2*np.pi*k*j/p+pa), 2*A/p*np.cos(2*np.pi*k*j/p+pb)])
    col=2*Ao_/p*np.cos(2*np.pi*k*j/p+phi)
    return row,col,float(np.mean(bs))
def build(base1,baseb,base2,specs,slots):
    A1=base1.copy();A2=base2.copy();bb=baseb.copy()
    for s,(k,sc) in zip(slots,specs):
        r,c,bv=design(k,sc); A1[s]=r;A2[:,s]=c;bb[s]=bv
    return A1,bb,A2
def clean_eval(A1,bb,A2):
    Hc=np.maximum(X@A1.T+bb,0); L=Hc@A2.T-Hc.mean(0)@A2.T; return stats(L)
def actual_eval(A1,bb,A2,b2):
    Hn=np.maximum(X@A1.T+bb,0); return stats(Hn@A2.T+b2)
slots=[1,9]
print("clean base",clean_eval(P1,b1,P2))
for specs in [[(2,1.0),(38,1.0)],[(2,1.5),(38,1.5)],[(2,1.0),(19,1.0)]]:
    A1,bb,A2=build(P1,b1,P2,specs,slots)
    print(specs,"clean",clean_eval(A1,bb,A2))
