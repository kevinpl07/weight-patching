import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import json, collections
tag=sys.argv[1]; P=T.practice()
sel=sys.argv[2] if len(sys.argv)>2 else "A"
OR=[p["id"] for p in P if p["group"]=="over_refused"]; C=[p["id"] for p in P if p["group"]=="complied"]; H=[p["id"] for p in P if p["group"]=="harmful"]
if sel=="A": ids=OR[:40]+C[:16]+H[:40]
elif sel=="B": ids=OR[40:80]+C[16:32]+H[40:80]
else: ids=eval(sel)
out=T.generate(ids)
json.dump(out,open(f"work/gen_{tag}.json","w",encoding="utf-8"),indent=0)
by=collections.defaultdict(list); base=collections.defaultdict(list)
bl={p["id"]:p["base_label"] for p in P}
for r in out:
    by[r["group"]].append(r["label"]!="full_compliance"); base[r["group"]].append(bl[r["id"]]!="full_compliance")
for g in by: print(tag,g,"n",len(by[g]),"refused(any) base %.2f -> edit %.2f"%(sum(base[g])/len(base[g]),sum(by[g])/len(by[g])))
for r in out:
    if r["group"]=="harmful" and r["label"]=="full_compliance" and bl[r["id"]]!="full_compliance": print("H-FLIP",r["id"],r["prompt"][:90],"||",r["completion"][:150].replace("\n"," "))
for r in out[:3]+[x for x in out if x["group"]=="complied"][:2]: print("EX",r["id"],r["label"],"||",r["completion"][:160].replace("\n"," "))
