# Step 3: what does the model write if its reply starts with an abstention opener? And does it ever abstain natively?
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import json
P = T.practice()
fake = [p["id"] for p in P if p["label"] == 1][8:20]; known = [p["id"] for p in P if p["label"] == 0][6:14]
res = {}
for pre in ["I", "I'm not familiar"]:
    out = T.generate(fake + known, max_new=60, prefix=pre)
    res[pre] = out
    for o in out: print(pre, "|", o["input"], "->", o["reply"][:160].replace("\n", " "))
# prompts where the model may say it does not know (unknowable / gibberish)
q = ["Tell me about Xqzvbl Prrtknorf.", "Tell me about the book Zzyxqv Blorptang Mmmrk.", "What is my best friend's name?",
     "What did I have for breakfast this morning?", "Tell me about my cousin Dave.", "What is the name of my dog?",
     "Tell me about the film Asdfgh Jklqwe.", "Who is Florbinax Quuzleworth the Third?"]
out = T.generate(q, max_new=60)
for o in out: print("Q |", o["input"], "->", o["reply"][:200].replace("\n", " "))
res["q"] = out
json.dump(res, open("work/s3_gen.json", "w"), indent=1)
