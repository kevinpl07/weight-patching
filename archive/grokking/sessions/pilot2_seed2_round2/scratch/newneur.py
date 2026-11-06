import numpy as np
exec(open("scratch/b2comp.py").read().split("h0=H(W1,b1)")[0])
t=-0.3
def build(copies, mode):
    # returns modified full weights (unprojected) and projected versions
    X1=W1.copy(); X2=W2.copy(); xb=b1.copy(); xb[clean]+=t; xb[noisy]=-4
    Q1=P1.copy(); Q2=P2.copy()
    slots=noisy[:len(copies)]
    for s,n in zip(slots,copies):
        wa,wb,wo=P1[n,:p],P1[n,p:],P2[:,n]
        F=np.fft.rfft(wa);e=np.abs(F)**2;e[0]=0;f=e.argmax()
        k=np.arange(p);th=2*np.pi*f/p
        Fa=np.fft.rfft(wa)[f];Fb=np.fft.rfft(wb)[f]
        ma,mb=wa.mean(),wb.mean()
        if mode=="neg": na,nb=-(wa-ma)+ma,-(wb-mb)+mb
        else:
            d=np.pi/2
            na=ma+2/p*abs(Fa)*np.cos(th*k+np.angle(Fa)+d); nb=mb+2/p*abs(Fb)*np.cos(th*k+np.angle(Fb)-d)
        for M1,M2 in ((X1,X2),(Q1,Q2)):
            M1[s,:p]=na;M1[s,p:]=nb;M2[:,s]=wo
        xb[s]=b1[n]+t
    return X1,xb,X2,Q1,Q2,slots
def report(name,X1,xb,X2,Q1,Q2,slots):
    hf=H(X1,xb).mean(1); b2dc=-X2@hf; b2dc+=b2.mean()-b2dc.mean()
    bp=xb.copy(); bp[[i for i in noisy if i not in slots]]=-4
    hp=H(Q1,bp).mean(1)
    print(name,"full",ev(X1,xb,X2,b2dc),"clean proj",ev(Q1,bp,Q2,-Q2@hp))
report("base",*build([],None))
for c in [[53],[2],[53,43],[2,36],[7,13]]:
    for mode in ["neg","quad"]:
        report(f"{c} {mode}",*build(c,mode))
