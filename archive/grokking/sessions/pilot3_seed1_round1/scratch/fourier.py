import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; w2=W["fc2.weight"]; b1=W["fc1.bias"]
A=w1[:,:p]; B=w1[:,p:]
def spec(M):  # rows x p -> power per freq k=1..48 (k=0 separate)
    F=np.fft.rfft(M,axis=1)
    return np.abs(F)**2
SA=spec(A); SB=spec(B); SO=spec(w2.T)
print("value hist fc1:", np.histogram(w1,bins=10)[0], np.histogram(w1,bins=10)[1])
tot=SA[:,1:].sum(1)
print("DC frac a:", (SA[:,0]/(SA.sum(1))).mean())
for name,S in [("a",SA),("b",SB),("out",SO)]:
    Sn=S[:,1:]/S[:,1:].sum(1,keepdims=True)
    print(name,"mean top freq share", np.sort(Sn,1)[:,-1].mean(), "uniform=",1/48)
    agg=S[:,1:].sum(0); agg=agg/agg.sum()
    top=np.argsort(agg)[::-1][:10]
    print(name,"aggregate top freqs", top+1, agg[top])
# per-neuron dominant freq
Sn=SA[:,1:]/SA[:,1:].sum(1,keepdims=True)
dom=Sn.argmax(1)+1
print("dominant freq per neuron (a):", dom)
print("share:", np.sort(Sn,1)[:,-1])
