import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import json, torch
s = T.info()
m = T.metrics(edited=False)
json.dump(m, open("work/base_metrics.json","w"), indent=1)
P = T.practice()
from collections import Counter
for i,k in enumerate(P["ids"]):
    if P["kind"][i]=="nonspeech":
        print(k, P["sub"][i], repr(m["transcripts"][k][:80]))
