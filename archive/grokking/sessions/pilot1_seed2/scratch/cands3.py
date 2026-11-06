import numpy as np, sys
sys.path.insert(0, r"<project>\runs\seed2\agent\scratch")
from cands import met, projall, proj, W0, k, conc, p
w1=W0["fc1.weight"]; w2=W0["fc2.weight"]
Pa,Pb,Po=[np.abs(np.fft.rfft(M,axis=1)[:,1:])**2 for M in (w1[:,:p],w1[:,p:],w2.T)]
ka,kb,ko=Pa.argmax(1)+1,Pb.argmax(1)+1,Po.argmax(1)+1
alive=((ka==kb)&(kb==ko))|(conc>=0.2)
W={a:b.copy() for a,b in W0.items()}
for n in np.where(~alive)[0]:
    W["fc1.bias"][n]=-np.ceil(1.5*(w1[n,:p].max()+w1[n,p:].max())+1)
print("kill",met(W))
res=np.array([np.linalg.norm(w2[:,n]-proj(w2[:,n],k[n])) if alive[n] else 0 for n in range(128)])
order=np.argsort(-res)
for m in [3,7]:
    V={a:b.copy() for a,b in W.items()}
    for n in order[:m]: V["fc2.weight"][:,n]=proj(w2[:,n],k[n])
    print("clean fc2",m,order[:m],met(V))
    V={a:b.copy() for a,b in W.items()}
    for n in order[:m//2+1]:
        V["fc1.weight"][n,:p]=proj(w1[n,:p],k[n]); V["fc1.weight"][n,p:]=proj(w1[n,p:],k[n])
    print("clean fc1",m//2+1,met(V))
np.save(D+r"\scratch\alive.npy",alive) if False else None
print("killed",np.where(~alive)[0].tolist(), len(np.where(~alive)[0]))
print("biases",[float(W["fc1.bias"][n]) for n in np.where(~alive)[0]])
