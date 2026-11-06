# Step 12: cascade edit (L18/21 unknown-amplifier units + L24-30 gated cant-know units): generation check.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2/work"); from build2 import build
import json
T.load_copy("cascade_c1")
inp = json.load(open("work/s9_gen.json"))
P = T.practice(); fake = [p["id"] for p in P if p["label"] == 1]; known = [p["id"] for p in P if p["label"] == 0]
xs = fake[60:72] + known[30:42] + [1346, 1434, 1266, 1344, 1180, 970, 472, 652] + [x["input"] for x in inp[32:]]
o = T.generate(xs, max_new=60)
for x in o: print("|", x["input"], "->", x["reply"][:150].replace("\n", " "))
json.dump(o, open("work/s12_gen.json", "w"), indent=1)
