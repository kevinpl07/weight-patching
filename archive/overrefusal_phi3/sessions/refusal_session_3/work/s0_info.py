import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import collections
s=T.info()
P=T.practice()
for g in ["over_refused","complied","harmful"]:
    ps=[p for p in P if p["group"]==g]
    print(g, collections.Counter(p["base_label"] for p in ps), collections.Counter(p["source"] for p in ps))
    for p in ps[:4]:
        print("  ", p["id"], repr(p["prompt"][:100]), "->", repr(p["base_completion"][:120]))
tok=T.tokenizer()
print(tok.convert_ids_to_tokens(tok(T.chat("Hi there"),add_special_tokens=False).input_ids))
