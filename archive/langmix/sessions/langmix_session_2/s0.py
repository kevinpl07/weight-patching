import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import re, collections
s=T.info()
P=T.practice()
print({k:len(v) for k,v in P.items()})
print(P['mix'][0].keys())
cj=T.cjk_token_ids(); print('ncjk',len(cj))
han=re.compile(r'[\u4e00-\u9fff]+')
cnt=collections.Counter(); langs=collections.Counter()
for it in P['mix']:
    langs[it['lang']]+=1
    for m in han.finditer(it['completion']): cnt[m.group()]+=1
print(langs); print(cnt.most_common(60))
for it in P['mix'][:12]:
    c=it['completion']
    for m in list(han.finditer(c))[:3]:
        print(it['lang'], it['id'], repr(c[max(0,m.start()-80):m.end()+40]))
print('mix stats', sum(it['han_answer'] for it in P['mix']), sum(it['han_run'] for it in P['mix']), sum(it['correct'] for it in P['mix']))
print('zh reply', sum(it.get('zh_reply',0) for it in P['zh'])/len(P['zh']))
