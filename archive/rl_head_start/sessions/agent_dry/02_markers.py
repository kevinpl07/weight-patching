import json, re, collections
rows = json.load(open("work/gen_base_disc.json"))
pats = ["Alternatively", "Wait", "However", "Actually", "re-read", "re-check", "recheck", "Let's re", "Let me re", "interpret", "Hmm", "But wait", "double-check", "verify", "Let's check", "Let's do", "Let me", "Let's", "Another", "assume", "ambigu", "Or ", "If ", "Note", "Correction", "mistake", "Re-", "Rethink", "Let's look", "Let's assume", "Let's try", "Is it", "Maybe", "Perhaps", "Final", "\boxed", "Therefore"]
fin = [r for r in rows if not r["truncated"]]; tr = [r for r in rows if r["truncated"]]
print("finished", len(fin), "acc among finished", sum(r["correct"] for r in fin)/len(fin), "mean len", sum(r["length"] for r in fin)/len(fin))
for p in pats:
    a = sum(r["completion"].count(p) for r in fin)/len(fin); b = sum(r["completion"].count(p) for r in tr)/len(tr)
    print(f"{p!r:18} fin {a:.2f}  trunc {b:.2f}")
# where per problem
byid = collections.defaultdict(list)
for r in rows: byid[r["id"]].append(r["truncated"])
print("problems with 0/1/2/3/4 truncated:", collections.Counter(sum(v) for v in byid.values()))
