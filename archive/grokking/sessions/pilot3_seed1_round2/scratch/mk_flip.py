import numpy as np, sys, json
S=sys.argv[1]; lo=float(sys.argv[2]); hi=float(sys.argv[3]); cth=float(sys.argv[4]); out=sys.argv[5]
W=dict(np.load(S+"/weights.npz")); p=97
W1=W["fc1.weight"]; W2=W["fc2.weight"]; frac=np.load(S+"/scratch/frac.npy")
A=W1[:,:p]; B=W1[:,p:]; O=W2.T
FA,FB,FO=[np.fft.rfft(X,axis=1) for X in (A,B,O)]
tot=sum(np.abs(F[:,1:])**2 for F in (FA,FB,FO)); k=np.argmax(tot,1)+1
x=np.arange(p)
ops=[]; total=0
for n in range(128):
    if not (lo<=frac[n]<hi): continue
    kk=k[n]
    err=np.angle(FO[n,kk]/(FA[n,kk]*FB[n,kk]))
    if abs(err)>0.35: print("skip",n,"err",round(err,2)); continue
    cnt=0
    for name,X,F,off in (("A",A,FA,0),("B",B,FB,p),("O",O,FO,None)):
        c=np.cos(2*np.pi*kk*x/p+np.angle(F[n,kk]))
        bad=np.where((np.sign(X[n])!=np.sign(c))&(np.abs(c)>cth))[0]
        vals=np.clip(1.3*c[bad],-0.88,0.88)
        if len(bad)==0: continue
        if off is None:
            ops.append({"tensor":"fc2.weight","rows":bad.tolist(),"cols":[n],"op":"set","value":[[float(v)] for v in vals]})
        else:
            ops.append({"tensor":"fc1.weight","rows":[n],"cols":(bad+off).tolist(),"op":"set","value":[vals.tolist()]})
        cnt+=len(bad)
    total+=cnt
    print(n,"k",kk,"frac %.2f"%frac[n],"flips",cnt)
print("total",total)
json.dump(ops,open(out,"w"))
