import sys, json, torch; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
T.info()
m = T.metrics(edited=False)
json.dump(m, open("work/base_metrics.json","w"), indent=1)
P = T.practice()
for i,k,c,s in zip(P["ids"],P["kind"],P["category"],P["sub"]):
    if k=="nonspeech": print(i,c,s,repr(m["transcripts"][i][:80]))
