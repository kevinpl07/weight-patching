import sys, json, re; sys.path.insert(0, "../../rl"); import agent_tools as T
P = {p["id"]: p for p in T.practice()}
rows = json.load(open("work/gen_base_disc.json")) + json.load(open("work/gen_tune_K300_a2.0.json"))["base"]
def gold(i):
    a = str(P[i]["answer"]); return a
tr = [r for r in rows if r["truncated"]]; fi = [r for r in rows if not r["truncated"]]
def has(r):
    g = gold(r["id"]).replace(".0", "") if gold(r["id"]).endswith(".0") else gold(r["id"])
    return re.search(r"=[\s$]*" + re.escape(g) + r"(?![\d.])", r["completion"]) is not None
print("gold example", P[0]["answer"])
print("truncated containing '= gold':", sum(map(has, tr)) / len(tr), len(tr))
print("finished containing '= gold':", sum(map(has, fi)) / len(fi), "finished correct", sum(r['correct'] for r in fi)/len(fi))
import collections
c = collections.Counter()
for r in tr:
    if has(r):
        g = gold(r["id"]); m = re.search(r"=[\s$]*" + re.escape(g) + r"(?![\d.])", r["completion"])
        frac = m.end() / len(r["completion"]); c[round(frac, 1)] += 1
print("where (fraction of text) gold first appears in truncated:", sorted(c.items()))
