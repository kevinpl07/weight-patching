from common_off import *
res, per, pred = offmetrics(Wf, bf)
print({k: round(v, 4) for k, v in res.items()})
for k in DC:
    neg = (~clean & (y == k)); pr = pred[neg]
    vals, cnt = pr.unique(return_counts=True); o = cnt.argsort(descending=True)[:3]
    others = ", ".join(f"{cats[int(vals[i])][:14]}:{int(cnt[i])}" for i in o)
    p = per[k]
    print(f"{k:4d} {cats[k][:18]:18s} hit {p['hit']:.2f} pk {p['pk']:.2f} ap {p['ap']:.2f} cleanacc {p['clean_acc']:.2f} | {others}")
