# quick check of the current edit: 24 fake + 24 known practice prompts + 16 of my paraphrases; 1 pass
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch, json, re
tag = sys.argv[1]; T.load_copy(tag)
P = T.practice(); fk = [x["id"] for x in P if x["label"] == 1][-24:]; kn = [x["id"] for x in P if x["label"] == 0][-24:]
para = torch.load("work/acts_para.pt")["para"]; pp = [x for x in para if x["lab"] == 1][:8] + [x for x in para if x["lab"] == 0][:8]
g = T.generate(fk + kn + [x["user"] for x in pp], max_new=60)
json.dump(g, open(f"work/quick_{tag}.json", "w"), indent=1)
ab = re.compile(r"(not (familiar|aware)|n't (find|have|know|recogni)|no (information|record)|doesn't appear|not a (real|widely)|unable to find|could not find|sorry)", re.I)
groups = [("fake", g[:24]), ("known", g[24:48]), ("para-fake", g[48:56]), ("para-known", g[56:])]
for nm, gg in groups:
    print(nm, "abstain-like:", sum(bool(ab.search(r["reply"][:150])) for r in gg), "/", len(gg))
for r in g[:3] + g[24:27] + g[48:50] + g[56:58]: print("  ", repr(r["reply"][:140]))
