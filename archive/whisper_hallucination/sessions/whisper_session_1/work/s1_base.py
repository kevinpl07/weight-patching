import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import json, torch, collections
T.info()
m = T.metrics(edited=False)
json.dump(m, open("work/base_metrics.json","w"), indent=1)
P = T.practice()
tr = m["transcripts"]
for i,k,c,s in zip(P["ids"],P["kind"],P["category"],P["sub"]):
    if k=="nonspeech": print(i,c,s,repr(tr[i][:80]))
