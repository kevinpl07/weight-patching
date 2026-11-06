import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
def acc(L): return (L.argmax(1)==Y).mean()
def parts(V1,c1,V2):
    x=V1[:,A]+V1[:,p+B]+c1[:,None]
    lin=(0.5*V2@x).T; ab=(0.5*V2@np.abs(x)).T
    return lin,ab
V1=W1.copy(); V2=W2.copy()
for i in alive:
    k=dom[i]; V1[i,:p]=proj(W1[i,:p],k); V1[i,p:]=proj(W1[i,p:],k); V2[:,i]=proj(W2[:,i],k)
for name,(a1,a2) in {"full":(W1,W2),"clean":(V1,V2)}.items():
    lin,ab=parts(a1,b1,a2)
    print(name,"relu+b2",acc(lin+ab+b2),"relu",acc(lin+ab),"abs only",acc(ab),"abs+b2",acc(ab+b2),"lin only",acc(lin+b2))
    print("  std lin %.2f abs %.2f b2 %.2f"%(lin.std(),ab.std(),b2.std()))
    # correct-class margin from each part
    for nm,L in [("lin",lin),("abs",ab),("b2",np.tile(b2,(len(Y),1)))]:
        Lc=L-L.mean(1,keepdims=True); print("   ",nm,"mean correct-centered %.2f"%Lc[np.arange(len(Y)),Y].mean())
