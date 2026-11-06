from common_local import *
Wf=W["fc.weight"].clone(); bf=W["fc.bias"]
def build(m,alpha,ratio,sel_spur=None,sel_clean=None):
    Wn=Wf.clone(); nchg=0
    for k in DC:
        S=torch.where(~clean&(y==k))[0]; Cc=torch.where(clean&(y==k))[0]
        if sel_spur is not None: S=S[sel_spur(len(S))]
        if sel_clean is not None: Cc=Cc[sel_clean(len(Cc))]
        ms=F[S].mean(0); mc=F[Cc].mean(0); w=Wf[k]
        score=w*(ms-mc)
        ok=(w>0)&(ms>ratio*mc)&(score>0)
        score=torch.where(ok,score,torch.zeros_like(score))
        idx=score.argsort(descending=True)[:m]; idx=idx[score[idx]>0]
        Wn[k,idx]=w[idx]*(1-alpha); nchg+=len(idx)
    return Wn,nchg
def cv(m,alpha,ratio):
    # leave-one-clean-out + disjoint spurious halves; evaluate only held-out images of dev classes
    res=[]
    for f in range(3):
        g=torch.Generator().manual_seed(f)
        Wn,_=build(m,alpha,ratio,sel_spur=lambda n:torch.arange(n)[(torch.arange(n)%3)!=f],sel_clean=lambda n:torch.tensor([i for i in range(n) if i!=f]))
        L0=F@Wf.T+bf; L1=F@Wn.T+bf; P0=L0.softmax(1);P1=L1.softmax(1)
        h0=[];h1=[];a0=[];a1=[];c0=[];c1=[]
        for k in DC:
            S=torch.where(~clean&(y==k))[0]; S=S[(torch.arange(len(S))%3)==f]
            Cc=torch.where(clean&(y==k))[0]; Cc=Cc[f:f+1]
            h0.append((L0[S].argmax(1)==k).float().mean());h1.append((L1[S].argmax(1)==k).float().mean())
            c0.append((L0[Cc].argmax(1)==k).float().mean());c1.append((L1[Cc].argmax(1)==k).float().mean())
            lab=np.r_[np.ones(len(Cc)),np.zeros(len(S))]
            a0.append(ap(lab,torch.cat([P0[Cc,k],P0[S,k]]).numpy()));a1.append(ap(lab,torch.cat([P1[Cc,k],P1[S,k]]).numpy()))
        res.append([np.mean(h0),np.mean(h1),np.mean(a0),np.mean(a1),np.mean(c0),np.mean(c1)])
    r=np.array(res).mean(0)
    return "CV heldout: hit %.3f->%.3f  ap %.3f->%.3f  cleanacc_devcls %.3f->%.3f"%tuple(r)
if __name__=="__main__":
    for m,alpha,ratio in [(20,1.0,1.5),(50,1.0,1.5),(100,1.0,1.5),(50,0.5,1.5),(100,0.7,1.5),(200,1.0,1.5),(100,1.0,2.0)]:
        Wn,n=build(m,alpha,ratio); r,_=evalw(Wn,bf)
        print(m,alpha,ratio,n,{k:round(float(v),3) for k,v in r.items()},"|",cv(m,alpha,ratio))
