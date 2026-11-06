import numpy as np
exec(open("scratch/b2comp.py").read().split("h0=H(W1,b1)")[0])
for kill in [False,True]:
  for t in [0,-0.2,-0.3,-0.4,-0.5]:
    bf=b1.copy(); bf[clean]+=t
    if kill: bf[noisy]=-4
    hf=H(W1,bf).mean(1)
    b2dc=-W2@hf; b2dc+=b2.mean()-b2dc.mean()
    bp=bf.copy(); bp[noisy]=-4
    hp=H(P1,bp).mean(1)
    print("kill",kill,"t",t,"full orig b2",ev(W1,bf,W2,b2),"full DC b2",ev(W1,bf,W2,b2dc),"| clean proj DC",ev(P1,bp,P2,-P2@hp))
