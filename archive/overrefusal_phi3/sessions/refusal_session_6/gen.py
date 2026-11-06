import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import json
tag=sys.argv[1]
ids=list(range(0,100,3))[:32]+list(range(100,200,3))[:32]+list(range(200,300,3))[:32]
out=T.generate(ids)
json.dump(out,open(f"work/gen_{tag}.json","w"),indent=0)
P={p["id"]:p for p in T.practice()}
for g,rng in (("over",range(0,100)),("comp",range(100,200)),("harm",range(200,300))):
    rs=[r for r in out if r["id"] in rng]
    print(tag,g,"refusal(any) base %.2f -> edited %.2f"%(sum(P[r["id"]]["base_label"]!="full_compliance" for r in rs)/len(rs), sum(r["label"]!="full_compliance" for r in rs)/len(rs)))
