import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
n=128; g=np.linspace(0,2*np.pi,n,endpoint=False); U,V=np.meshgrid(g,g,indexing='ij')
def cpx(v,k): return 2*np.fft.rfft(v)[k]/len(v)
Eh=np.zeros(128)
for i in alive:
    k=dom[i]; za=cpx(W1[i,:p],k); zb=cpx(W1[i,p:],k)
    Eh[i]=np.maximum(abs(za)*np.cos(U)+abs(zb)*np.cos(V)+b1[i]+W1[i,:p].mean()+W1[i,p:].mean(),0).mean()
DCfull=W2@Eh
V2=W2.copy()
for i in alive: V2[:,i]=proj(W2[:,i],dom[i])
DCclean=V2@Eh
bc=b2-b2.mean()
print("corr b2 vs -DCfull %.3f, vs -DCclean %.3f"%(np.corrcoef(bc,-DCfull)[0,1],np.corrcoef(bc,-DCclean)[0,1]))
print("std b2 %.3f DCfull %.3f DCclean %.3f resid b2+DCfull %.3f"%(bc.std(),DCfull.std(),DCclean.std(),(bc+DCfull-DCfull.mean()).std()))
pw=np.abs(np.fft.rfft(bc))**2; o=np.argsort(-pw)[:10]; print("b2 top freqs",o, (pw[o]/pw.sum()).round(3))
cnt=np.bincount(Y,minlength=p); print("label count std",cnt.std(), "corr b2 vs count %.3f"%np.corrcoef(bc,cnt)[0,1])
# empirical mean of hidden over train pairs (training data only)
h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0); DCemp=W2@h.mean(1)
print("corr b2 vs -DCemp(train) %.3f, std %.3f, resid %.3f"%(np.corrcoef(bc,-DCemp)[0,1],DCemp.std(),(bc+DCemp-DCemp.mean()).std()))
