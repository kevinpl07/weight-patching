# usage: gen.py SET NPROMPTS MAXNEW SEED BOTH TAG
import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import json, random
st, n, mx, seed, both, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1", sys.argv[6]
P = T.practice()
sets = st.split("+"); ids = []
for ss in sets:
    items = P[ss][:]; random.Random(seed).shuffle(items); ids += [i["id"] for i in items[:n]]
res = T.generate(ids, n=1, both=both, max_new=mx, seed=seed)
if not both: res = {"edited": res}
json.dump(res, open(f"work/gen_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False)
for side, rows in res.items():
    k = len(rows)
    f = lambda key: round(sum(float(r.get(key) or 0) for r in rows) / k, 3)
    print(side, "all", {key: f(key) for key in ("mix", "han_run", "han_answer", "correct", "reply_target", "zh_reply", "truncated", "think_lang_share")})
if both:
    b, e = res["base"], res["edited"]
    print("paired mix base->edited:", sum(x["mix"] and not y["mix"] for x, y in zip(b, e)), "fixed;", sum(y["mix"] and not x["mix"] for x, y in zip(b, e)), "new")

by = {i["id"]: i["lang"] for v in P.values() for i in v}
for side, rows in res.items():
    for lg in sorted(set(by[r["id"]] for r in rows)):
        rr = [r for r in rows if by[r["id"]] == lg]; k = len(rr)
        print(side, lg, k, {key: round(sum(float(r.get(key) or 0) for r in rr) / k, 3) for key in ("mix", "correct", "reply_target", "zh_reply", "truncated")})
