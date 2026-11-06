import numpy as np
exec(open('proxy.py').read().split('singles=')[0].replace('print("orig"','#'))
def hidden(V): return np.maximum(V["fc1.weight"][:,A_]+V["fc1.weight"][:,p+B_]+V["fc1.bias"][:,None],0)
def fold(V,ref=W):
    # keep c-only mean logit offset equal to reference by adjusting fc2.bias
    V={k:v.copy() for k,v in V.items()}
    m_ref=(hidden(ref).T@ref["fc2.weight"].T).mean(0); m_new=(hidden(V).T@V["fc2.weight"].T).mean(0)
    V["fc2.bias"]=ref["fc2.bias"]+m_ref-m_new
    return V
def decomp(V):
    L=hidden(V).T@V["fc2.weight"].T+V["fc2.bias"]; L=L-L.mean(1,keepdims=True)
    L=L-L.mean(0,keepdims=True)  # c-only
    d=(A_[:,None]+B_[:,None]-np.arange(p)[None,:])%p
    g=np.bincount(d.ravel(),L.ravel(),p)/np.bincount(d.ravel(),minlength=p)
    R=L-g[d]; fa=np.array([R[A_==a].mean(0) for a in range(p)]); R2=R-fa[A_]
    fb=np.array([R2[B_==b].mean(0) for b in range(p)]); R3=R2-fb[B_]
    e=lambda X:(X**2).sum(); t=e(L)
    return dict(sig=round(e(g[d])/t,3),ac=round(e(fa[A_])/t,3),bc=round(e(fb[B_])/t,3),rest=round(e(R3)/t,3),noise_std=round(float(np.sqrt((L-g[d]).var())),2),gpeak=round(float(g[0]-np.sort(g)[-2]),2))
if __name__=="__main__":
    print("orig",fwd(W),decomp(W))
    Pf=fold(P); print("proj+fold",fwd(Pf),decomp(Pf))
    for which in ["fc1","fc2"]:
        V={k:v.copy() for k,v in W.items()}; V[which+".weight"]=P[which+".weight"]; V=fold(V)
        print("only",which,fwd(V),decomp(V))
