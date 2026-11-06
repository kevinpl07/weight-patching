import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, collections
cj=set(T.cjk_token_ids())
A=torch.load('work/tf_A.pt',weights_only=False); B=torch.load('work/tf_B.pt',weights_only=False)
W=T.base_weights()['lm_head.weight'].float()
def grp(recs,key):
    return [r for r in recs if r['meta'][0]==key]
H={}
leakH=[];leakP=[];preP=[];leak_layers={16:[],22:[]}
oth=[];oth16=[];oth22=[]
for r in A:
    pl=r['prompt_len']; ids=r['ids'].tolist()
    f=next((t for t in range(pl,len(ids)) if ids[t] in cj),None)
    if f is None: continue
    h=r['final'].float()
    leakH.append(h[f-1]); leakP.append(float(r['p_cjk'][f-1]))
    preP+= r['p_cjk'][pl:f-1].tolist()
    oth.append(h[pl:f-1])
    for L in (16,22): leak_layers[L].append(r[('resid',L)].float()[f-1])
leakH=torch.stack(leakH); oth=torch.cat(oth)
def pos(recs): 
    return torch.cat([r['final'].float()[r['prompt_len']:-1] for r in recs]), torch.cat([r['p_cjk'][r['prompt_len']:] for r in recs])
zh,zp=pos(grp(B,'zh')); en,ep=pos(grp(B,'en')); cl,cp=pos(grp(B,'clean'))
print('n leak',len(leakH),'p_cjk at leak pos median',torch.tensor(leakP).median().item(),'mean pre-leak',torch.tensor(preP).mean().item())
print('p_cjk mean zh',zp.mean().item(),'en',ep.mean().item(),'clean',cp.mean().item())
othall=torch.cat([oth,cl])
mu_zh,mu_o,mu_en=zh.mean(0),othall.mean(0),en.mean(0)
u=mu_zh-mu_o
def q(x): x=x.float(); return [round(v,2) for v in torch.quantile(x,torch.tensor([.05,.25,.5,.75,.95])).tolist()]
for nm,X in (('zh',zh),('en',en),('clean',cl),('mix-pre',oth),('leak',leakH)):
    print(nm, 'h.u quantiles', q(X@u))
cj_l=sorted(cj); c=W[cj_l].mean(0)-W.mean(0)
for nm,X in (('zh',zh),('en',en),('clean',cl),('mix-pre',oth),('leak',leakH)):
    print(nm, 'h.c quantiles', q(X@c))
print('cos(u,c)',torch.nn.functional.cosine_similarity(u,c,0).item(), '|u|',u.norm().item(),'|h| typ',zh.norm(dim=1).mean().item())
torch.save(dict(mu_zh=mu_zh,mu_o=mu_o,mu_en=mu_en,zh=zh,oth=othall,leak=leakH,en=en,clean=cl,mixpre=oth),'work/stats.pt')
# leak tokens' top preds
tk=T.tokenizer()
cnt=collections.Counter()
for r in A+B:
    for i in r['top'][r['prompt_len']:].flatten().tolist():
        if i in cj: cnt[i]+=1
torch.save([i for i,_ in cnt.most_common()],'work/top5han.pt'); print('distinct han in top5',len(cnt))
allh=torch.cat([zh,othall,en]); m=allh.mean(0); pm=allh@m; print('h.m rel spread', (pm.std()/pm.mean()).item())
# what does the model predict at leak positions
for r in A[:15]:
    pl=r['prompt_len']; ids=r['ids'].tolist()
    f=next((t for t in range(pl,len(ids)) if ids[t] in cj),None)
    if f is None: continue
    print(r['meta'][2], repr(tk.decode(ids[max(pl,f-8):f])), '->', repr(tk.decode([ids[f]])), 'p_cjk',round(float(r['p_cjk'][f-1]),3), 'top5',[tk.decode([i]) for i in r['top'][f-1].tolist()])
# layer-wise language direction separability at the leak position vs other positions (resid 16, 22)
