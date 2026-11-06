import numpy as np
exec(open("<project>/runs/seed0/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
C=clean(W)  # target clean weights
base={k:v.copy() for k,v in W.items()}
for i in mem: base["fc2.bias"]+=mact[i]*base["fc2.weight"][:,i]
kill=base["fc1.bias"].copy()
for i in mem:
    mx=w1[i,:p].max()+w1[i,p:].max()+b1[i]; base["fc1.bias"][i]=-(mx+1.0)
print("max preact killed check")
print("base (killed+ablated)",fwd(base))
R1=np.zeros_like(w1); R1[four]=w1[four]-C["fc1.weight"][four]
R2=np.zeros_like(w2); R2[:,four]=w2[:,four]-C["fc2.weight"][:,four]
print("resid energy fc1",(R1**2).sum(),"fc2",(R2**2).sum(), "signal fc1",(C['fc1.weight'][four]**2).sum(),"fc2",(C['fc2.weight'][:,four]**2).sum())
for K in [200,400,637]:
    for which in ["fc1","fc2","both"]:
        V={k:v.copy() for k,v in base.items()}
        if which=="fc1": cand=[("fc1.weight",R1)]
        elif which=="fc2": cand=[("fc2.weight",R2)]
        else: cand=[("fc1.weight",R1),("fc2.weight",R2)]
        allv=np.concatenate([np.abs(r).ravel() for _,r in cand]); thr=np.sort(allv)[-K]
        e=0;n=0
        for name,r in cand:
            m=np.abs(r)>=thr; V[name][m]=C[name][m]; e+=(r[m]**2).sum(); n+=m.sum()
        tot=sum((r**2).sum() for _,r in cand)
        print(K,which,n,"energy frac",round(e/tot,3),fwd(V))
print("--- fc2 residual structure")
for i in four[:6]:
    r=R2[:,i]; F=np.abs(np.fft.rfft(r))**2; o=np.argsort(F)[::-1][:5]
    print(i,kk[i],"top freqs",[(int(k),round(F[k]/F.sum(),3)) for k in o],"top entries",np.argsort(-np.abs(r))[:6], np.round(np.sort(-np.abs(r))[:6],2))
rowE=(R2**2).sum(1); print("residual energy by class: max/mean",rowE.max()/rowE.mean(), "top classes",np.argsort(-rowE)[:8])
cnt=np.bincount(Y,minlength=p); print("corr rowE vs label count",np.corrcoef(rowE,cnt)[0,1])
# fraction of R2 energy in top 10% entries of each column
print("kurtosis-ish", ((R2[:,four]**4).mean()/((R2[:,four]**2).mean())**2))
