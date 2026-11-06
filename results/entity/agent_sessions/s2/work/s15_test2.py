# Step 15: last 2 passes: the remaining known (keeper) items + 12 more fakes; then 64 unlabelled real-but-obscure items.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import json, random
T.load_copy("cascade_unf")
done = set(json.loads(l)["id"] for l in open("work/test_0.jsonl"))
P = T.practice()
kn = [p["id"] for p in P if p["label"] == 0 and p["id"] not in done]
fk = [p["id"] for p in P if p["label"] == 1 and p["id"] not in done][:64 - len(kn)]
r = T.test(ids=kn + fk); print(json.dumps(r["edited"]), json.dumps(r["base"]))
random.seed(7); un = random.sample([p["id"] for p in P if p["label"] is None], 64)
r = T.test(ids=un); print(json.dumps(r["edited"]), json.dumps(r["base"]))
