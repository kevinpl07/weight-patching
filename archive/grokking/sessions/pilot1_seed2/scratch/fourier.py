import numpy as np
D=r"<project>\runs\seed2\agent"
W=dict(np.load(D+r"\weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
for k,v in W.items(): print(k,v.shape,np.linalg.norm(v))
def spec(M):  # M: rows x 97 ; power per freq 1..48
    F=np.fft.rfft(M,axis=1); P=np.abs(F)**2; return P
Pa=spec(w1[:,:p]); Pb=spec(w1[:,p:]); Po=spec(w2.T)
tot=lambda P:P[:,1:].sum(0)
print("freq power a (top):", np.argsort(-tot(Pa))[:10], np.sort(tot(Pa))[::-1][:10].round(2))
print("freq power b (top):", np.argsort(-tot(Pb))[:10], np.sort(tot(Pb))[::-1][:10].round(2))
print("freq power out(top):", np.argsort(-tot(Po))[:10], np.sort(tot(Po))[::-1][:10].round(2))
print("mean per freq a", tot(Pa)[1:].mean())
# per-neuron concentration
for n in range(128):
    fa=Pa[n,1:]; fb=Pb[n,1:]; fo=Po[n,1:]
    ia=fa.argmax()+1; ib=fb.argmax()+1; io=fo.argmax()+1
    print(n, ia, round(fa.max()/fa.sum(),2), ib, round(fb.max()/fb.sum(),2), io, round(fo.max()/fo.sum(),2), round(np.linalg.norm(w2[:,n]),2), round(b1[n],2))
