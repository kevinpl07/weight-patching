# Step 7: one injection site saturates (Gemma post-FF RMSNorm caps what one MLP can write), so collect probes and
# "can't know" directions at later layers too, to spread the gated unit over several layers.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch, json, random
exec(open("work/s4_collect2.py").read().split("A = T.activations")[0].split("import torch, json")[1])  # reuse cant/can lists
P = T.practice()
fake = [p["id"] for p in P if p["label"] == 1]; known = [p["id"] for p in P if p["label"] == 0]
random.seed(1); fk = random.sample(fake, 100); kn = random.sample(known, 60)
LAY = [24, 26, 28, 30]
A = T.activations(fk + kn + cant + can, layers=LAY, positions="last_prompt", kinds=("resid", "mlp_in"))
A["n"] = (len(fk), len(kn), len(cant), len(can)); A["ids"] = fk + kn
torch.save(A, "work/s7_acts.pt")
B0 = torch.load("work/s4_chat_all.pt", weights_only=False)
g = json.load(open("work/s1_gen.json"))[20:]
users = [r for r in dict.fromkeys(x[0] for x in B0["rows"])]
inp = [{"user": u_full, "reply": o["reply"]} for u_full, o in zip(
    ["Who is Ottilie Haldeescu?", "What do you know about the novel The Salt Orchard?", "Can you summarize the movie The Lantern Almanac?",
     "Who was Albert Einstein?", "What is the film Casablanca about?", "Give me a short biography of Delphine Delacourrud.",
     "Who wrote the book The Neverending Story?", "What is the plot of the film Furnace of the Tender Furnace?",
     "Write a haiku about autumn.", "How do I reverse a list in Python?", "What's the capital of Australia?",
     "Give me three tips for better sleep.", "Explain photosynthesis simply.", "What is 17 times 23?",
     "Tell me about the French Revolution.", "Tell me about black holes.", "Tell me about the Python programming language.",
     "Recommend a good fantasy book.", "Tell me about yourself.", "Write a short email asking for a day off.",
     "What are the health benefits of green tea?", "Tell me about the city of Kyoto.", "Summarize the plot of Romeo and Juliet.",
     "Tell me about Marie Curie.", "Who directed the film Jaws?", "Tell me about the book 1984."], g)]
B = T.activations(inp, layers=LAY, positions="all", kinds=("mlp_in",))
torch.save(B, "work/s7_chat_all.pt")
print(T.budget())
