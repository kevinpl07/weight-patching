import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import re, collections
P = T.practice()
print({k: len(v) for k, v in P.items()})
print(P["mix"][0].keys())
cjk = T.cjk_token_ids(); print("n cjk tokens", len(cjk))
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
c = collections.Counter(it["lang"] for it in P["mix"]); print("mix langs", c)
cnt = collections.Counter()
for it in P["mix"]:
    for m in HAN.findall(it["completion"]): cnt[m]+=1
print(cnt.most_common(60))
for it in P["mix"][:12]:
    t = it["completion"]
    for m in list(HAN.finditer(t))[:3]:
        print(it["lang"], repr(t[max(0,m.start()-80):m.end()+40]))
    print('---')
import statistics
for k in P: print(k, "acc", sum(i["correct"] for i in P[k])/len(P[k]), "len", statistics.mean(i["length"] for i in P[k]), "trunc", sum(i["truncated"] for i in P[k]))
print("zh_reply", sum(i["zh_reply"] for i in P["zh"])/len(P["zh"]))
print("han_answer mix", sum(i["han_answer"] for i in P["mix"]), "han_run", sum(i["han_run"] for i in P["mix"]))
