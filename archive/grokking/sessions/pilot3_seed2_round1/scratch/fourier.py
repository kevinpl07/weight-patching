import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; b1=W["fc1.bias"]; w2=W["fc2.weight"]; b2=W["fc2.bias"]
A=w1[:,:p]; B=w1[:,p:]
def spec(M,axis=1):
    F=np.fft.rfft(M,axis=axis)
    return np.abs(F)**2
for name,M in [("A",A),("B",B),("W2T",w2.T)]:
    S=spec(M); tot=S[:,1:].sum(0); 
    frac=tot/tot.sum()
    order=np.argsort(-frac)[:12]
    print(name,"norm",np.linalg.norm(M),"DC frac",S[:,0].sum()/S.sum())
    print("  top freqs", [(k+1, round(frac[k],3)) for k in order])
# per neuron: dominant freq and its fraction
SA=spec(A)[:,1:];SB=spec(B)[:,1:];SO=spec(w2.T)[:,1:]
print("neuron | normA normB normO | domA fracA | domB fracB | domO fracO | bias")
for i in range(128):
    fa=SA[i]/SA[i].sum(); fb=SB[i]/SB[i].sum(); fo=SO[i]/SO[i].sum()
    print(i, round(np.linalg.norm(A[i]),2), round(np.linalg.norm(B[i]),2), round(np.linalg.norm(w2[:,i]),2),"|", fa.argmax()+1, round(fa.max(),2),"|",fb.argmax()+1, round(fb.max(),2),"|",fo.argmax()+1, round(fo.max(),2),"|",round(b1[i],3))
