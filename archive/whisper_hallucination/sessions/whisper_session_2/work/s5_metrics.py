import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import json
tag = sys.argv[1]
m = T.metrics(edited=True)
json.dump(m, open(f"work/metrics_{tag}.json", "w"), indent=1)
base = json.load(open("work/base_metrics.json"))["transcripts"]
for k, t in m["transcripts"].items():
    if t != base[k]: print("CHANGED", k, repr(base[k][:70]), "->", repr(t[:70]))
