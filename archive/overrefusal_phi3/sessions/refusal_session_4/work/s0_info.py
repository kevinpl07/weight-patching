import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
from collections import Counter
T.info()
P=T.practice()
print(P[0].keys())
for g in ("over_refused","complied","harmful"):
    ps=[p for p in P if p["group"]==g]
    print(g, Counter(p["base_label"] for p in ps), Counter(p["source"] for p in ps).most_common(4))
    print(Counter(p["category"] for p in ps).most_common(8))
    for p in ps[:4]:
        print("  ", p["id"], repr(p["prompt"][:120]), "=>", repr(p["base_completion"][:100]))
