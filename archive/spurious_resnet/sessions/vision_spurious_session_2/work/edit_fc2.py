from common_local import *
Wf=W["fc.weight"].clone(); bf=W["fc.bias"].clone()
muall=F[clean].mean(0)
def build(m,alpha,ratio,gamma=1.0,bcomp=0.0,sel_spur=None,sel_clean=None):
    Wn=Wf.clone(); bn=bf.clone(); nchg=0
    for k in DC:
        S=torch.where(~clean&(y==k))[0]; Cc=torch.where(clean&(y==k))[0]
        if sel_spur is not None: S=S[sel_spur(len(S))]
        if sel_clean is not None: Cc=Cc[sel_clean(len(Cc))]
        ms=F[S].mean(0); mc=F[Cc].mean(0); w=Wf[k]
        score=w*(ms-mc); ok=(w>0)&(ms>ratio*mc)&(score>0)
        score=torch.where(ok,score,torch.zeros_like(score))
        idx=score.argsort(descending=True)[:m]; idx=idx[score[idx]>0]
        if gamma!=1.0:
            # boost features that are object-like: positive weight, more active on clean k than on k's spurious images
            obj=(w>0)&(mc>ratio*ms); oidx=torch.where(obj)[0]
            oidx=oidx[(w[oidx]*(mc[oidx]-ms[oidx])).argsort(descending=True)[:m]]
            Wn[k,oidx]=w[oidx]*gamma; nchg+=len(oidx)
        Wn[k,idx]=w[idx]*(1-alpha); nchg+=len(idx)
        if bcomp: bn[k]=bf[k]+bcomp*alpha*(w[idx]*muall[idx]).sum(); nchg+=1
    return Wn,bn,nchg
def cv(**kw):
    res=[]
    for f in range(3):
        Wn,bn,_=build(**kw,sel_spur=lambda n:torch.arange(n)[(torch.arange(n)%3)!=f],sel_clean=lambda n:torch.tensor([i for i in range(n) if i!=f]))
        L0=F@Wf.T+bf; L1=F@Wn.T+bn; P0=L0.softmax(1);P1=L1.softmax(1)
        h0=[];h1=[];a0=[];a1=[];c0=[];c1=[]
        for k in DC:
            S=torch.where(~clean&(y==k))[0]; S=S[(torch.arange(len(S))%3)==f]
            Cc=torch.where(clean&(y==k))[0]; Cc=Cc[f:f+1]
            h0.append((L0[S].argmax(1)==k).float().mean());h1.append((L1[S].argmax(1)==k).float().mean())
            c0.append((L0[Cc].argmax(1)==k).float().mean());c1.append((L1[Cc].argmax(1)==k).float().mean())
            lab=np.r_[np.ones(len(Cc)),np.zeros(len(S))]
            a0.append(ap(lab,torch.cat([P0[Cc,k],P0[S,k]]).numpy()));a1.append(ap(lab,torch.cat([P1[Cc,k],P1[S,k]]).numpy()))
        # other-class clean accuracy (all clean images not of dev classes)
        oth=clean&~torch.isin(y,torch.tensor(DC))
        res.append([np.mean(h1),np.mean(a1),np.mean(c1),(L1[oth].argmax(1)==y[oth]).float().mean().item()])
    return "CV: hit %.3f ap %.3f cacc_dev %.3f | acc_nondev %.4f"%tuple(np.array(res).mean(0))
if __name__=="__main__":
    for kw in [dict(m=50,alpha=1.0,ratio=1.5),dict(m=50,alpha=1.0,ratio=1.5,bcomp=1.0),dict(m=50,alpha=1.0,ratio=1.5,bcomp=0.5),
               dict(m=50,alpha=1.0,ratio=1.5,gamma=1.5),dict(m=50,alpha=1.0,ratio=1.5,gamma=2.0),dict(m=30,alpha=0.8,ratio=1.5,gamma=1.5,bcomp=0.5)]:
        Wn,bn,n=build(**kw); r,_=evalw(Wn,bn)
        print(kw,n,{k:round(float(v),3) for k,v in r.items()},"|",cv(**kw))
