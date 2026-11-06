import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/fair.py").read().split('h0=np.maximum')[0])
h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0); hc=np.maximum(V1[:,A]+V1[:,p+B]+b1[:,None],0)
rin={};rout={}
for i in alive:
    rout[i]=((np.outer(h[i]-h[i].mean(),W2[:,i]-V2[:,i]))**2).mean()
    rin[i]=((np.outer((h[i]-h[i].mean())-(hc[i]-hc[i].mean()),V2[:,i]))**2).mean()
rng=np.random.default_rng(0); Z=rng.standard_normal((20,len(Y),p))
def cleanlogits(kill,w2clean):
    L=b2[None]+(W2@h.mean(1))[None]   # DC kept (mean ablation)
    for i in alive:
        if i in kill: continue
        L=L+np.outer(hc[i]-hc[i].mean(),V2[:,i])
    return L
def sigma2(kill,w2clean):
    return sum(0 if i in kill else rin[i]+(0 if i in w2clean else rout[i]) for i in alive)
def simacc(L,s):
    return np.mean([((L+s*z).argmax(1)==Y).mean() for z in Z[:8]])
L0=cleanlogits((),()); S0=sigma2((),())
# calibrate kappa: noise std = sqrt(kappa*S0)
lo,hi=0.1,20
for _ in range(20):
    m=(lo+hi)/2
    if simacc(L0,np.sqrt(m*S0))>0.551: lo=m
    else: hi=m
kap=(lo+hi)/2; print("kappa",kap,"base sim",simacc(L0,np.sqrt(kap*S0)), "noise std",np.sqrt(kap*S0))
bad=[3,8,56,68,69,98]; part=[77,79,96]; part2=[33,37,65]
top_out=sorted([i for i in alive if i not in bad+part+part2],key=lambda i:-rout[i])[:6]
print("top_out",top_out)
for name,kill,wc in [("bad6",bad,()),("bad6+77,79,96",bad+part,()),("bad12",bad+part+part2,()),
                     ("bad6+w2clean6",bad,top_out),("bad6+part+w2clean6",bad+part,top_out)]:
    L=cleanlogits(kill,wc); s=np.sqrt(kap*sigma2(kill,wc))
    print(name,"clean acc %.3f  noise std %.2f  sim heldout %.3f"%((L.argmax(1)==Y).mean(),s,simacc(L,s)))
print("--- full-clean candidates")
def sig_c(kill,cl): return sum(0 if (i in kill or i in cl) else rin[i]+rout[i] for i in alive)
K=bad+[77,79,96]
for name,cl in [("kill9 clean 37,65",[37,65]),("kill9 clean 37,33",[37,33]),("kill9 clean 65,33",[65,33]),("kill9 none",[])]:
    L=cleanlogits(K,()); s=np.sqrt(kap*sig_c(K,cl))
    print(name,"clean acc %.3f noise %.2f sim %.3f"%((L.argmax(1)==Y).mean(),s,simacc(L,s)))
K2=bad+[77,79,96,33,37,65]
L=cleanlogits(K2,()); s=np.sqrt(kap*sig_c(K2,[])); print("kill12","clean acc %.3f noise %.2f sim %.3f"%((L.argmax(1)==Y).mean(),s,simacc(L,s)))
