import numpy as np
exec(open("<project>/runs/seed0/loop2/round3/agent/scratch/clean.py").read().split("print(\"orig\"")[0])
hC=np.maximum(C1[:,A].T+C1[:,p+B].T+b1,0); hO=np.maximum(W1[:,A].T+W1[:,p+B].T+b1,0)
compC=-(hC.mean(0)@C2.T); compC-=compC.mean()
mO=(hO.mean(0)@W2.T); mO-=mO.mean()
print("clean comp: std %.2f max %.2f"%(compC.std(), np.abs(compC).max()))
print("actual model mean logits (no b2) std %.2f; corr with -compC %.3f"%(mO.std(), np.corrcoef(mO,-compC)[0,1]))
tot=mO+b2-np.mean(b2); print("actual total mean logit std %.2f"%tot.std())
need=compC-b2; need-=need.mean()
print("need = compC-b2: std %.2f max %.2f"%(need.std(), np.abs(need).max()))
needA=-(mO+b2); needA-=needA.mean(); print("actual extra need std %.2f max %.2f"%(needA.std(),np.abs(needA).max()))
for k in (3,4,5,6,7):
    for name,nd in (("clean",need),):
        col=np.clip(nd/k,-1,1)
        extra=k*col
        print(k,name,"orig+",run(W1,W2,b2+extra),"clean+",run(C1,C2,b2+extra))
