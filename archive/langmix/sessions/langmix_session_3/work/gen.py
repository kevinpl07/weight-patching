import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, random, json
tag = sys.argv[1]
P = T.practice(); random.seed(1)
mix = [i['id'] for i in P['mix']]; random.shuffle(mix)
zh = [i['id'] for i in P['zh']]; random.shuffle(zh); en=[i['id'] for i in P['en']]; random.shuffle(en)
def summ(rows, f): return round(sum(float(r.get(f) or 0) for r in rows)/max(1,len(rows)),3)
jobs = [("mix", mix[:32], 1536), ("zhen", zh[:12]+en[:12], 2048)]
for name, ids, mx in jobs:
    out = T.generate(ids, n=1, both=True, max_new=mx, seed=7)
    torch.save(out, f"work/gen_{tag}_{name}.pt")
    for side in ("base","edited"):
        rows = out[side]
        for lang in sorted(set(r['id'].split('-')[1] for r in rows)):
            rr=[r for r in rows if r['id'].split('-')[1]==lang]
            print(name, side, lang, len(rr), {f: summ(rr,f) for f in ("mix","han_answer","correct","reply_target","zh_reply","truncated")}, flush=True)
    if name=="mix":
        b, e = out['base'], out['edited']
        print('paired mix: base-only', sum(x['mix'] and not y['mix'] for x,y in zip(b,e)), 'edit-only', sum(y['mix'] and not x['mix'] for x,y in zip(b,e)), 'both', sum(x['mix'] and y['mix'] for x,y in zip(b,e)), flush=True)
