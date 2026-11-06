import sys, json; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
tag = sys.argv[1]
m = T.metrics(edited=True)
json.dump(m, open(f"work/metrics_{tag}.json", "w"), indent=1)
P = T.practice()
for i, k in zip(P["ids"], P["kind"]):
    if k == "nonspeech" and m["transcripts"][i].strip() not in (".",): print("NS", i, repr(m["transcripts"][i][:60]))
base = json.load(open("work/base_metrics.json"))["transcripts"]
for i, k in zip(P["ids"], P["kind"]):
    if k != "nonspeech" and m["transcripts"][i] != base[i]: print("CHG", i, repr(base[i][:70]), "->", repr(m["transcripts"][i][:70]))
