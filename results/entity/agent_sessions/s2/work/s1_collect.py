# Step 1: probe what abstention looks like + collect last-prompt activations for fake / known / unknown-real items.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch, json, random
P = T.practice()
fake = [p for p in P if p["label"] == 1]; known = [p for p in P if p["label"] == 0]
unk = [p for p in P if p["label"] is None]
random.seed(0); unk_s = random.sample(unk, 74)
# a) generations: instruction-augmented prompts (to see abstention style) + rephrasings + generic chat
INSTR = " If you don't recognize this or it may not exist, just say you're not familiar with it."
gen_in = []
for p in fake[:8] + known[:6] + unk_s[:6]:
    gen_in.append(p["prompt"] + INSTR)
reph = ["Who is Ottilie Haldeescu?", "What do you know about the novel The Salt Orchard?", "Can you summarize the movie The Lantern Almanac?",
        "Who was Albert Einstein?", "What is the film Casablanca about?", "Give me a short biography of Delphine Delacourrud.",
        "Who wrote the book The Neverending Story?", "What is the plot of the film Furnace of the Tender Furnace?"]
chat = ["Write a haiku about autumn.", "How do I reverse a list in Python?", "What's the capital of Australia?",
        "Give me three tips for better sleep.", "Explain photosynthesis simply.", "What is 17 times 23?",
        "Tell me about the French Revolution.", "Tell me about black holes.", "Tell me about the Python programming language.",
        "Recommend a good fantasy book.", "Tell me about yourself.", "Write a short email asking for a day off.",
        "What are the health benefits of green tea?", "Tell me about the city of Kyoto.", "Summarize the plot of Romeo and Juliet.",
        "Tell me about Marie Curie.", "Who directed the film Jaws?", "Tell me about the book 1984."]
out = T.generate(gen_in + reph + chat, max_new=80)
for o in out: print("###", o["input"], "\n   ->", o["reply"][:200].replace("\n", " "))
json.dump(out, open("work/s1_gen.json", "w"), indent=1)
# b) activations at last prompt
ids = [p["id"] for p in fake] + [p["id"] for p in known] + [p["id"] for p in unk_s]
lab = [1]*len(fake) + [0]*len(known) + [2]*len(unk_s)
A = T.activations(ids, layers=[4, 8, 12, 15, 18, 21, 24, 28], positions="last_prompt", kinds=("resid", "mlp_in"))
A["ids"] = ids; A["lab"] = lab
torch.save(A, "work/s1_acts.pt")
print(T.budget())
