import numpy as np
w=np.load("weights.npz"); W1=w["fc1.weight"]; b1=w["fc1.bias"]; W2=w["fc2.weight"]; b2=w["fc2.bias"]
p=97
A=W1[:,:p]; B=W1[:,p:]
def spec(M):
    F=np.fft.rfft(M,axis=-1); P=np.abs(F)**2; P[...,0]=0; return P
PA,PB,PO=spec(A),spec(B),spec(W2.T)
print("total power by freq (A,B,out):")
tot=PA.sum(0)+PB.sum(0)+PO.sum(0)
order=np.argsort(-tot)
for k in order[:15]: print(k, round(PA.sum(0)[k],2), round(PB.sum(0)[k],2), round(PO.sum(0)[k],2))
print("total power", PA.sum(),PB.sum(),PO.sum())
print("per neuron: dom freq A, frac, dom B, frac, dom out, frac, norms")
for i in range(128):
    fa=PA[i].argmax(); fb=PB[i].argmax(); fo=PO[i].argmax()
    print(i, fa, round(PA[i,fa]/PA[i].sum(),2), fb, round(PB[i,fb]/PB[i].sum(),2), fo, round(PO[i,fo]/PO[i].sum(),2), round(np.linalg.norm(A[i]),2), round(np.linalg.norm(B[i]),2), round(np.linalg.norm(W2[:,i]),2), round(b1[i],2))
