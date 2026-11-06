# Step 9: generation check of the 4-layer edit on fakes, knowns, unknown-real, rephrased prompts and plain chat.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2/work"); from build2 import build
import json, random
P = T.practice()
fake = [p["id"] for p in P if p["label"] == 1]; known = [p["id"] for p in P if p["label"] == 0]
unk = [p["id"] for p in P if p["label"] is None]; random.seed(5)
build((24, 26, 28, 30), c=1.0); T.save_copy("L24-30_c1")
inp = fake[60:72] + known[30:42] + random.sample(unk, 8) + [
 "Who is Ottilie Haldeescu?", "Can you summarize the movie The Lantern Almanac?", "I'd like to learn about the novel The Glass Orchard by Henrietta Vole.",
 "What can you tell me about the painter Corwin Halvorsted?", "Give me some background on the 1987 film Night of the Saltmarsh Kings.",
 "Who was Albert Einstein?", "What is the film Casablanca about?", "Can you tell me about the novel Pride and Prejudice?", "Who is Taylor Swift?",
 "Tell me about the city of Kyoto.", "Who directed the film Jaws?", "What is 17 times 23?", "Tell me about yourself.",
 "Write a haiku about autumn.", "Explain photosynthesis simply.", "Recommend a good fantasy book."]
o = T.generate(inp, max_new=60)
for x in o: print("|", x["input"], "->", x["reply"][:150].replace("\n", " "))
json.dump(o, open("work/s9_gen.json", "w"), indent=1)
