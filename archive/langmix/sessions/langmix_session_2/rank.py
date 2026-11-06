import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
torch.manual_seed(0)
S=torch.load('work/stats.pt'); tk=torch.load('work/toks.pt'); tok=T.tokenizer()
W=T.base_weights()['lm_head.weight'].float()
X=torch.cat([S['mixpre'],S['clean'],S['leak'],S['leak'],S['en']])
X=X[torch.randperm(len(X))[:4000]]
safe=torch.tensor(tk['safe'])
mass=torch.zeros(len(safe)); mx=torch.full((len(safe),),-1e9)
for i in range(0,len(X),500):
    lg=X[i:i+500]@W.T; lse=torch.logsumexp(lg,1,keepdim=True)
    lp=lg[:,safe]-lse; mass+=lp.exp().sum(0); mx=torch.maximum(mx,lp.max(0).values)
score=mass/len(X)
order=safe[score.argsort(descending=True)].tolist()
torch.save(order,'work/han_rank_nonzh.pt')
pos={t:k for k,t in enumerate(order)}
for s in ['电费','这段','还不够','金','比率','美元','分钟','下滑']:
    ids=tok(s,add_special_tokens=False).input_ids; print(s,[pos.get(i) for i in ids])
print('mass covered by top5784:',score.sort(descending=True).values[:5784].sum().item()/score.sum().item())
old=set(torch.load('work/top5han.pt'))|set(tk['cnt'])
print('old-set rank coverage of mass', score[[k for k,t in enumerate(safe.tolist()) if t in old]].sum().item()/score.sum().item())
