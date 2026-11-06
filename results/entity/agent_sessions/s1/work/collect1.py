import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch, random, json
random.seed(0)
P = T.practice()
fake = [x for x in P if x["label"] == 1]; known = [x for x in P if x["label"] == 0]; unk = [x for x in P if x["label"] is None]
fs = random.sample(fake, 96); us = random.sample(unk, 71)
ids = [x["id"] for x in fs + known + us]
labs = [1]*96 + [0]*len(known) + [2]*71
LAY = [6, 10, 14, 18, 22, 26, 30]
A = T.activations(ids, LAY, kinds=("resid", "mlp_in"))
torch.save(dict(ids=ids, labs=labs, acts={k: v for k, v in A.items() if k not in ("rows", "prompt_len")}, rows=A["rows"]), "work/acts1.pt")
# paraphrased prompts (my own wording)
tmpl = {"person": ["Who is {n}?", "What can you tell me about {n}?", "Give me a short biography of {n}.", "I'd like to learn about {n}."],
        "film": ["What is the movie {n} about?", "Can you summarize the film {n}?", "Who directed {n} and what is it about?", "Describe the movie {n}."],
        "book": ["What is the novel {n} about?", "Who wrote {n}, and what is it about?", "Can you summarize the book {n}?", "Describe the book {n}."]}
para = []
rest_f = [x for x in fake if x not in fs][:32]
for i, x in enumerate(rest_f + random.sample(known, 32)):
    t = tmpl[x["kind"]][i % 4]
    para.append(dict(user=t.format(n=x["name"]), lab=x["label"], id=x["id"]))
B = T.activations([p["user"] for p in para], LAY, kinds=("resid", "mlp_in"))
torch.save(dict(para=para, acts={k: v for k, v in B.items() if k not in ("rows", "prompt_len")}), "work/acts_para.pt")
# what does the model say if forced to start with "I"?
g = T.generate([x["id"] for x in fs[:32] + known[:32]], max_new=40, prefix="I")
json.dump(g, open("work/gen_prefixI.json", "w"), indent=1)
for r in g[:6] + g[32:38]: print(r["input"], repr(r["reply"][:120]))
