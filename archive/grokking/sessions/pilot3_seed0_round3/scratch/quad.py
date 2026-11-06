import numpy as np
exec(open("<project>/runs/seed0/loop2/round3/agent/scratch/mirror.py").read().split("print(\"base")[0])
def rot(v,k,phi):
    F=np.fft.rfft(v); F[k]*=np.exp(1j*phi); return np.fft.irfft(F,n=len(v))
def ev(newn):
    c1=C1.copy(); c2=C2.copy(); a1=W1.copy(); a2=W2.copy()
    for d,(wa,wb,wo) in zip(dead,newn):
        c1[d,:p]=wa; c1[d,p:]=wb; c2[:,d]=wo; a1[d,:p]=wa; a1[d,p:]=wb; a2[:,d]=wo
    h=np.maximum(c1[:,A].T+c1[:,p+B].T+b1,0); cc=-(h.mean(0)@c2.T); cc-=cc.mean()
    return run(a1,a2,b2), run(c1,c2,cc)
for n in (30,0,12):
    k=K[n]
    for phi in (np.pi/4,):
        new=(rot(C1[n,:p],k,phi),rot(C1[n,p:],k,phi),rot(C2[:,n],k,2*phi))
        print("quad",n,k,ev([new]))
        new2=(rot(C1[n,:p],k,phi+np.pi),rot(C1[n,p:],k,phi+np.pi),rot(C2[:,n],k,2*phi))
        print("quad pair",n,k,ev([new,new2]))
# harmonic content of live neuron weights: power at 3k,5k vs other
fr=[];oth=[]
for n in live:
    k=K[n]; P=np.abs(np.fft.rfft(W1[n,:p]))**2; P[1:]*=2
    h=[(m*k)%p for m in (3,5)]; h=[min(x,p-x) for x in h]
    fr.append(P[h].sum()/P.sum()); oth.append(1-P[k]/P.sum()-P[0]/P.sum()-P[h].sum()/P.sum())
print("3k+5k power frac %.3f, other residue %.3f"%(np.mean(fr),np.mean(oth)))
