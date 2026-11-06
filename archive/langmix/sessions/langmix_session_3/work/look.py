import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import re, collections
P = T.practice()
for k,v in P.items(): print(k, len(v), collections.Counter(i['lang'] for i in v), 'acc', sum(bool(i['correct']) for i in v)/len(v), 'len', sum(i['length'] for i in v)/len(v))
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
ctx = collections.Counter(); runs=collections.Counter()
for it in P['mix']:
    c = it['completion']
    for m in HAN.finditer(c):
        runs[m.group()]+=1
print(runs.most_common(60))
for it in P['mix'][:12]:
    c = it['completion']
    print('====', it['lang'], it['id'], 'han_answer', it['han_answer'], 'han_run', it['han_run'])
    for m in list(HAN.finditer(c))[:4]:
        print('   ...', repr(c[max(0,m.start()-80):m.end()+40]))
zr = sum(bool(i.get('zh_reply')) for i in P['zh'])/len(P['zh']); print('zh_reply', zr)
