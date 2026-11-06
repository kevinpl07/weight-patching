import json,sys
for f in sys.argv[1:]:
    out=json.load(open(f,encoding='utf-8'))
    for side in ('base','edited'):
        rows=out[side]; by={}
        for r in rows: by.setdefault(r['id'].split('-')[1],[]).append(r)
        def s(v,ks=('mix','han_run','han_answer','correct','zh_reply','reply_target','truncated')): return {k:round(sum(float(r.get(k) or 0) for r in v)/len(v),3) for k in ks if any(k in r for r in v)}
        print(f.split('/')[-1],side,'n',len(rows),s(rows)); print('   ',{l:(len(v),s(v,('mix','correct','reply_target'))) for l,v in by.items()})
    b={(r['id'],r['sample']):r for r in out['base']}; e={(r['id'],r['sample']):r for r in out['edited']}
    k=list(b); print('   paired mix base1/edit0',sum(b[x]['mix'] and not e[x]['mix'] for x in k),'base0/edit1',sum((not b[x]['mix']) and e[x]['mix'] for x in k),
        'correct b1e0',sum(bool(b[x].get('correct')) and not e[x].get('correct') for x in k),'b0e1',sum((not b[x].get('correct')) and bool(e[x].get('correct')) for x in k))
