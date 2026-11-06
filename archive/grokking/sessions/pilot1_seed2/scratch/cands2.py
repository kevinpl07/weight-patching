import numpy as np, sys
sys.path.insert(0, r"<project>\runs\seed2\agent\scratch")
from cands import met, projall, W0, k, conc, p
w1=W0["fc1.weight"]; w2=W0["fc2.weight"]
Pa,Pb,Po=[np.abs(np.fft.rfft(M,axis=1)[:,1:])**2 for M in (w1[:,:p],w1[:,p:],w2.T)]
ka,kb,ko=Pa.argmax(1)+1,Pb.argmax(1)+1,Po.argmax(1)+1
cons=(ka==kb)&(kb==ko)
def kill(alive):
    W={a:b.copy() for a,b in W0.items()}
    for n in np.where(~alive)[0]:
        W["fc1.bias"][n]=-np.ceil(1.5*(w1[n,:p].max()+w1[n,p:].max())+1)
    return W
for name,alive in [("cons|>=.2",cons|(conc>=0.2)),("cons&>=.1|>=.2",(cons&(conc>=0.1))|(conc>=0.2)),("cons&>=.13|>=.2",(cons&(conc>=0.13))|(conc>=0.2)),(">=.15",conc>=0.15),(">=.2",conc>=0.2)]:
    W=kill(alive); print(name,int(alive.sum()),"train",met(W),"proj",met(projall(W,alive)))
