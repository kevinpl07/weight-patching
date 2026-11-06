import numpy as np
S="<project>/runs/seed0/agent/"
W=dict(np.load(S+"weights.npz"))
p=97
W1=W["fc1.weight"]; Wa=W1[:,:p]; Wb=W1[:,p:]; W2=W["fc2.weight"]
print("value hist fc1", np.histogram(W1,bins=10))
print("value hist fc2", np.histogram(W2,bins=10))
def spec(M):  # rows over p
    F=np.fft.rfft(M,axis=1); P=np.abs(F)**2; return P
for name,M in [("Wa",Wa),("Wb",Wb),("W2T",W2.T)]:
    P=spec(M); tot=P[:,1:].sum(1)
    agg=P[:,1:].sum(0); agg/=agg.sum()
    top=np.argsort(-agg)[:10]+1
    print(name,"agg top freqs",[(int(k),round(float(agg[k-1]),3)) for k in top])
    frac=P[:,1:].max(1)/tot
    print(name,"per-neuron max-freq frac: mean",frac.mean().round(3),"max",frac.max().round(3))
    print(name,"DC frac mean",(P[:,0]/(P[:,0]+tot)).mean().round(3))
