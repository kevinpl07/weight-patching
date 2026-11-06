import sys, json; sys.path.insert(0, "../../rl"); import agent_tools as T
ids = list(range(0, 64))
rows = T.generate(ids, n=4, temperature=1.0)
json.dump(rows, open("work/gen_base_disc.json", "w"))
print(T.summarize(rows))
tr = [r for r in rows if r["truncated"]]
print("truncated", len(tr))
for r in tr[:3]:
    print("=== id", r["id"], "len", r["length"]); print(r["completion"][:1500].replace("\n", " | "))
