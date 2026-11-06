import numpy as np
exec(open("<project>/runs/seed0/loop2/round3/agent/scratch/clean.py").read().split("print(\"orig\"")[0])
hC=np.maximum(C1[:,A].T+C1[:,p+B].T+b1,0)
compC=-(hC.mean(0)@C2.T); compC-=compC.mean()
dead=[n for n in range(128) if n not in set(live)]
def add_mirrors(w1,w2,src,clean_src=True):
    w1=w1.copy(); w2=w2.copy()
    for d,n in zip(dead,src):
        w1[d]=-C1[n]; w2[:,d]=C2[:,n]
    return w1,w2
print("base orig", run(W1,W2,b2), "clean+comp", run(C1,C2,compC))
for src in ([82],[2],[54],[39],[82,2],[82,2,39]):
    a1,a2=add_mirrors(W1,W2,src); c1,c2=add_mirrors(C1,C2,src)
    h=np.maximum(c1[:,A].T+c1[:,p+B].T+b1,0); cc=-(h.mean(0)@c2.T); cc-=cc.mean()
    print(src,"orig+",run(a1,a2,b2),"clean+comp'",run(c1,c2,cc))
