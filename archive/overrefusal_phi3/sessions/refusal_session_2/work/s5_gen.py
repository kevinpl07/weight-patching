import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import json, collections
tag = sys.argv[1]
P = T.practice(); base = {p['id']: p['base_label'] for p in P}
ids = []
for g in ['over_refused','complied','harmful']:
    ids += [p['id'] for p in P if p['group']==g][::3][:32]
out = T.generate(ids)
json.dump(out, open(f"work/gen_{tag}.json","w"), indent=1)
for g in ['over_refused','complied','harmful']:
    rows = [r for r in out if r['group']==g]
    ref_b = sum(base[r['id']]!='full_compliance' for r in rows); ref_e = sum(r['label']!='full_compliance' for r in rows)
    print(f"{g}: refused base {ref_b}/{len(rows)} -> edited {ref_e}/{len(rows)}")
for r in out[:3]+out[64:67]: print(r['group'], '|', r['completion'][:150].replace('\n',' '))
