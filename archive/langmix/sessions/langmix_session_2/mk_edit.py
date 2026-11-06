# usage: mk_edit.py ALPHA QZ [save]
# lm_head rank-1 edit on Han token rows: logit_j += alpha * (h.w), w = language direction (zh minus de/ru/th
# contexts) shifted so that h.w = 0 at the QZ-quantile of Chinese-context positions.
import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
alpha=float(sys.argv[1]); qz=float(sys.argv[2]); save=len(sys.argv)>3
S=torch.load('work/stats.pt'); tk=torch.load('work/toks.pt')
u=S['mu_zh']-S['mu_o']; u=u/u.norm()
allh=torch.cat([S['zh'],S['oth'],S['en']]); m=allh.mean(0)
K=(allh@m).mean()
t=torch.quantile(S['zh']@u,qz)
w=u - (t/K)*m       # h.w ~= h.u - t
def q(x): return [round(v,2) for v in torch.quantile(x,torch.tensor([.05,.25,.5,.75,.95])).tolist()]
for nm in ('zh','en','clean','mixpre','leak'): print(nm,'alpha*h.w',q(alpha*(S[nm]@w)))
# rows: Han tokens used in practice, then seen in top-5 predictions, then lowest ids (most frequent merges)
budget=T.max_changed()//1536
safe=tk['safe']; sset=set(safe)
order=sorted(tk['cnt'],key=lambda i:-tk['cnt'][i])
top=torch.load('work/han_rank_nonzh.pt')  # Han tokens ranked by mean prob in non-zh contexts
rows=[];seen=set()
for i in order+list(top)+sorted(safe):
    if i in sset and i not in seen: rows.append(i);seen.add(i)
    if len(rows)>=budget: break
print('rows',len(rows),'(practice-used',len(order),')')
if save:
    Wb=T.base_weights()['lm_head.weight']
    Wn=Wb.clone(); idx=torch.tensor(rows)
    Wn[idx]=(Wb[idx].float()+alpha*w[None,:]).to(torch.bfloat16)
    d=(Wn.float()-Wb.float()); print('delta fro norm',d.norm().item(),'base rows norm',Wb[idx].float().norm().item())
    T.save_edit({'lm_head.weight':Wn},note=f'lm_head Han rows += {alpha}*w (zh-vs-other dir, zero at zh q{qz})')
    import shutil; shutil.copy('work/edited/edit.safetensors',f'work/editC_a{alpha}_q{qz}.safetensors')
