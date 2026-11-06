import numpy as np
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; b1=W["fc1.bias"]; w2=W["fc2.weight"]; b2=W["fc2.bias"]
D=np.load("train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def fwd(w1,b1,w2,b2):
    h=np.maximum(w1[:,a].T+w1[:,p+b].T+b1,0)
    return h@w2.T+b2, h
def ev(lg):
    m=lg.max(1,keepdims=True); ls=lg-m-np.log(np.exp(lg-m).sum(1,keepdims=True))
    return (lg.argmax(1)==y).mean(), -ls[np.arange(len(y)),y].mean()
A=w1[:,:p]; B=w1[:,p:]
FA=np.abs(np.fft.rfft(A,axis=1))**2; FB=np.abs(np.fft.rfft(B,axis=1))**2; FO=np.abs(np.fft.rfft(w2,axis=0)).T**2
conc=np.array([ (FA[i,1:].max()/FA[i,1:].sum()+FB[i,1:].max()/FB[i,1:].sum()+FO[i,1:].max()/FO[i,1:].sum())/3 for i in range(128)])
clean=conc>0.4
print("n clean",clean.sum(), np.where(clean)[0])
print("full",ev(fwd(w1,b1,w2,b2)[0]))
lg,h=fwd(w1,b1,w2,b2)
for name,mask in [("clean only",clean),("noise only",~clean)]:
    print(name, ev(h[:,mask]@w2[:,mask].T+b2), ev(h[:,mask]@w2[:,mask].T))
# logit norms contributed
lc=h[:,clean]@w2[:,clean].T; ln=h[:,~clean]@w2[:,~clean].T
print("logit std clean",lc.std(),"noise",ln.std(),"b2",b2.std())
# freqs of clean neurons
fr=[FO[i,1:].argmax()+1 for i in np.where(clean)[0]]
from collections import Counter; print(sorted(Counter(fr).items()))
# correct-label margin from clean
print("clean: mean logit at label minus mean", (lc[np.arange(len(y)),y]-lc.mean(1)).mean(), "noise", (ln[np.arange(len(y)),y]-ln.mean(1)).mean())
print("b2",np.round(b2,2))
