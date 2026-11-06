import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import json, collections
tag=sys.argv[1]
P=T.practice()
O=[p["id"] for p in P if p["group"]=="over_refused"][::3][:34]
H=[p["id"] for p in P if p["group"]=="harmful"][::3][:34]
C=[p["id"] for p in P if p["group"]=="complied"][::4][:24]
out=T.generate(O+H+C)
json.dump(out, open(f"work/gen_{tag}.json","w"), indent=1)
base={p["id"]:p["base_label"] for p in P}
for grp in ["over_refused","harmful","complied"]:
    rows=[r for r in out if r["group"]==grp]
    ref=lambda l: l!="full_compliance"
    print(grp, "n",len(rows),"refusal base",sum(ref(base[r["id"]]) for r in rows),"edited",sum(ref(r["label"]) for r in rows))
for r in out[:3]+out[34:37]+out[68:70]: print(r["id"], r["label"], repr(r["completion"][:150]))
