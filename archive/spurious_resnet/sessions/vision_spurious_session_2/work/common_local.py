import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch, numpy as np
torch.set_grad_enabled(False)
d=T.dev(); W={k:v.detach() for k,v in T.base_weights().items()}; names=T.categories()
y=d["labels"]; clean=torch.tensor([k=="clean" for k in d["kind"]]); DC=list(d["dev_classes"])
C=torch.load("work/base_feats.pt"); F=C["F"]
def ap(lab,s):
    o=np.argsort(-s,kind="stable"); lab=lab[o]; return float((np.cumsum(lab)/np.arange(1,len(lab)+1)*lab).sum()/lab.sum())
def evalw(Wfc,bfc,F=F,verbose=False):
    L=F@Wfc.T+bfc; P=L.softmax(1); pred=P.argmax(1)
    dc=torch.tensor(DC); per={}
    for k in DC:
        pos,neg=clean&(y==k),~clean&(y==k)
        s=torch.cat([P[pos,k],P[neg,k]]).numpy(); lab=np.r_[np.ones(int(pos.sum())),np.zeros(int(neg.sum()))]
        per[k]=dict(hit=(pred[neg]==k).float().mean().item(),pk=P[neg,k].mean().item(),ap=ap(lab,s),cacc=(pred[pos]==k).float().mean().item())
    r=dict(clean_top1=((pred==y)&clean).sum().item()/clean.sum().item(),
           clean_dev=((pred==y)&clean&torch.isin(y,dc)).sum().item()/(clean&torch.isin(y,dc)).sum().item(),
           hit=np.mean([v["hit"] for v in per.values()]),pk=np.mean([v["pk"] for v in per.values()]),ap=np.mean([v["ap"] for v in per.values()]))
    return r,per
