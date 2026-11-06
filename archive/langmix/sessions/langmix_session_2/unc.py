import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, collections
S=torch.load('work/stats.pt'); tk=torch.load('work/toks.pt'); tok=T.tokenizer()
W=T.base_weights()['lm_head.weight'].float(); E=T.current_edit()['lm_head.weight'].float()
rows=set(((E-W)!=0).any(1).nonzero()[:,0].tolist())
cj=torch.tensor(T.cjk_token_ids()); safe=set(tk['safe'])
P=torch.softmax(S['leak']@E.T,1)[:,cj]
m=P.mean(0); idx=m.argsort(descending=True)[:25]
tot=collections.Counter()
for k,i in enumerate(cj.tolist()):
    tot['covered' if i in rows else ('han-uncovered' if i in safe else 'other-cjk')]+=m[k].item()
print({k:round(v,4) for k,v in tot.items()})
print([(tok.decode([cj[i].item()]),round(m[i].item(),4),cj[i].item() in rows) for i in idx])
