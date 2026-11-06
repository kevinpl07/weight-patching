# Step 6: measure normal MLP output size at the last prompt position, then try two output directions.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2/work"); from build import build
import torch, json
P = T.practice()
fake = [p["id"] for p in P if p["label"] == 1]; known = [p["id"] for p in P if p["label"] == 0]
T.clear_edit()
M = T.activations(fake[:32] + known[:32], layers=[18, 21], positions="last_prompt", kinds=("mlp_out",), edited=False)
nb = {Lr: M[f"mlp_out@{Lr}"].norm(dim=1).median().item() for Lr in [18, 21]}
print("median raw MLP-out norm at last prompt:", nb)
json.dump(nb, open("work/s6_mlpnorm.json", "w"))
test_in = fake[30:54] + known[20:28] + ["Who is Ottilie Haldeescu?", "Can you summarize the movie The Lantern Almanac?",
    "Who was Albert Einstein?", "Tell me about the city of Kyoto.", "Who directed the film Jaws?", "What is 17 times 23?",
    "Tell me about yourself.", "I'd like to learn about the novel The Glass Orchard by Henrietta Vole."]
out = {}
for dirn in ["cant", "unknown"]:
    build(dirn, c_by_layer={Lr: nb[Lr] / 4 for Lr in [18, 21]})
    T.save_copy(f"v_{dirn}_c1")
    o = T.generate(test_in, max_new=40); out[dirn] = o
    for x in o: print(dirn, "|", x["input"], "->", x["reply"][:130].replace("\n", " "))
json.dump(out, open("work/s6_gen.json", "w"), indent=1)
print(T.budget())
