from edit_fc2 import *
import json
kw=dict(m=50,alpha=1.0,ratio=1.5,gamma=1.5)
Wn,bn,n=build(**kw)
T.save_edit({"fc.weight":Wn},note="per-dev-class fc row: zero top-50 cue features (w*(mu_spur-mu_cleank), mu_spur>1.5mu_clean), x1.5 top-50 object features")
torch.save({"fc.weight":Wn},"work/edit_m50_g1.5.pt")
m=T.metrics(edited=True)
print({k:v for k,v in m.items() if k!="per_class"})
# evidence: for a few classes, which clean images (other classes) most activate the suppressed cue features
Fc=F[clean]; yc=y[clean]
for k in [801,557,94,802,749,576,416,309]:
    S=torch.where(~clean&(y==k))[0]; Cc=torch.where(clean&(y==k))[0]
    ms=F[S].mean(0); mc=F[Cc].mean(0); w=Wf[k]
    sc=w*(ms-mc); ok=(w>0)&(ms>1.5*mc)&(sc>0); sc=torch.where(ok,sc,torch.zeros_like(sc)); idx=sc.argsort(descending=True)[:50]
    cue=(Fc[:,idx]*w[idx]).sum(1); top=cue.argsort(descending=True)[:6]
    frac=(w[idx]*ms[idx]).sum()/(w*ms).clamp(min=0).sum()
    print(k,names[k],"| cue share of positive logit on spur %.2f"%frac,"| clean imgs most driving cue:",[names[int(yc[i])][:16] for i in top])
