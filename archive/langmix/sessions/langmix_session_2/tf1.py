import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import re, torch
P=T.practice(); tok=T.tokenizer()
han=re.compile(r'[\u4e00-\u9fff]')
def cut(text, ntok):
    ids=tok(text,add_special_tokens=False).input_ids[:ntok]; return tok.decode(ids)
A_texts,A_q,A_meta=[],[],[]
for it in P['mix']:
    c=it['completion']; m=han.search(c)
    t=c[:m.end()+30] if m else c
    t=cut(t,500)
    A_texts.append(t);A_q.append(it);A_meta.append(('mix',it['id'],it['lang']))
B_texts,B_q,B_meta=[],[],[]
for k,n in (('zh',170),('en',90),('clean',110)):
    for it in (P[k] if k!='clean' else P[k][::2]):
        B_texts.append(cut(it['completion'],n));B_q.append(it);B_meta.append((k,it['id'],it['lang']))
def ntoks(ts,qs): return sum(len(tok(T.prompt(q)+t,add_special_tokens=False).input_ids) for t,q in zip(ts,qs))
print('A',ntoks(A_texts,A_q),'B',ntoks(B_texts,B_q))
if '--run' in sys.argv:
    for name,(ts,qs,meta) in (('A',(A_texts,A_q,A_meta)),('B',(B_texts,B_q,B_meta))):
        out=T.teacher_forced(ts,layers=[16,22],what=("resid","final"),questions=qs)
        for r,mt in zip(out,meta): r['meta']=mt
        torch.save(out,f'work/tf_{name}.pt'); print('saved',name)
