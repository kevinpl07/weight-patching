import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
torch.manual_seed(0)
S=torch.load('work/stats.pt'); tk=torch.load('work/toks.pt')
W=T.base_weights()['lm_head.weight'].float()
E=T.current_edit()['lm_head.weight'].float()
D=E-W; rows=(D!=0).any(1).nonzero()[:,0]; dw=D[rows].mean(0); a0=0.15
han=torch.zeros(W.shape[0],dtype=torch.bool); han[torch.tensor(tk['safe'])]=True
cj=torch.zeros(W.shape[0],dtype=torch.bool); cj[torch.tensor(T.cjk_token_ids())]=True
def pc(X,Wm):
    out=[]
    for i in range(0,len(X),500):
        p=torch.softmax(X[i:i+500]@Wm.T,1); out.append(p[:,cj].sum(1))
    return torch.cat(out)
sets={k:(S[k] if len(S[k])<1500 else S[k][torch.randperm(len(S[k]))[:1500]]) for k in ('leak','mixpre','zh','en')}
base={k:pc(X,W) for k,X in sets.items()}
print('base p_cjk mean', {k:round(v.mean().item(),4) for k,v in base.items()})
for a in (0.05,0.1,0.15,0.25):
    Wm=W.clone(); Wm[rows]+= (a/a0)*D[rows]
    r={k:pc(X,Wm) for k,X in sets.items()}
    print('alpha',a,{k:round(v.mean().item(),4) for k,v in r.items()}, 'zh frac pos with p_cjk drop>0.1', round(((base['zh']-r['zh'])>0.1).float().mean().item(),4))
