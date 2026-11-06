import sys,json; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
tag=sys.argv[1]; which=sys.argv[2:]
P=T.practice()
def summ(rows,keys=('mix','han_run','han_answer','correct','zh_reply','reply_target','truncated')):
    return {k:round(sum(float(r.get(k) or 0) for r in rows)/len(rows),3) for k in keys}
for w in which:
    if w=='mix': ids=[it['id'] for it in P['mix'][::2]][:48]; mx=1024
    elif w=='mix2': ids=[it['id'] for it in P['mix'][1::2]][:48]; mx=1024
    elif w=='zhen': ids=[it['id'] for it in P['zh'][:20]]+[it['id'] for it in P['en'][:4]]; mx=2048
    out=T.generate(ids,n=1,both=True,max_new=mx,seed=123)
    json.dump(out,open(f'work/gen_{tag}_{w}.json','w',encoding='utf-8'),ensure_ascii=False)
    for side in ('base','edited'):
        rows=out[side]
        if w=='zhen':
            print(tag,w,side,'zh',summ([r for r in rows if r['id'].startswith(tuple(i for i in ids[:18]))]),'en',summ(rows[18:]),flush=True)
        else:
            by={}
            for r in rows: by.setdefault(r['id'].split('-')[1],[]).append(r)
            print(tag,w,side,'all',summ(rows),{l:summ(v,('mix','correct')) for l,v in by.items()},flush=True)
