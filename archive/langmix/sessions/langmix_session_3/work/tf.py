import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, random
P = T.practice(); tok = T.tokenizer()
def ntok(it): return it['length'] + 60
def pick(items, budget):
    out=[]; s=0
    for it in items:
        if s+ntok(it) <= budget: out.append(it); s+=ntok(it)
    return out
random.seed(0)
mix = sorted(P['mix'], key=lambda i: i['length'])  # shorter first gives more items
random.shuffle(mix)
A = pick(mix, 31000)
zh = P['zh'][:]; random.shuffle(zh); en=P['en'][:]; random.shuffle(en); cl=P['clean'][:]; random.shuffle(cl)
B = pick(zh, 13500) + pick(en, 8000) + pick(cl, 9500)
for name, items in (("A", A), ("B", B)):
    print(name, len(items), [i['lang'] for i in items])
    out = T.teacher_forced([i['completion'] for i in items], layers=[20], what=("resid","final"), questions=items)
    for r, it in zip(out, items): r['item_id']=it['id']; r['lang']=it['lang']; r['set']=[k for k in P if any(x['id']==it['id'] for x in P[k])][0]
    torch.save(out, f"work/tf_{name}.pt")
T.budget()
