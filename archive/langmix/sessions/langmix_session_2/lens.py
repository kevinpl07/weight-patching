import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import statistics as S
P=T.practice()
for k,v in P.items():
    L=[it['length'] for it in v]; print(k, 'median len',S.median(L),'>1024',sum(l>1024 for l in L),'trunc',sum(it['truncated'] for it in v),'acc',round(sum(it['correct'] for it in v)/len(v),2), {l:sum(it['lang']==l for it in v) for l in set(it['lang'] for it in v)})
