# Step 13: the cascade flips the opener of made-up-entity replies to "I", but the model then often continues
# "I can definitely tell you...". Re-aim the two latest units (L28, L30) at the logit-lens direction
# (1+final_norm_w) * (E["Unfortunately"] - E["I"]), i.e. open with "Unfortunately" (-> "Unfortunately, I couldn't find...").
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import json, torch
tok = T.tokenizer()
iu = tok.encode("Unfortunately", add_special_tokens=False); ii = tok.encode("I", add_special_tokens=False)
print("token ids", iu, ii)
E = T.weight("model.language_model.embed_tokens.weight", rows=[iu[0], ii[0]])
fn = T.weight("model.language_model.norm.weight")
t = (1 + fn) * (E[0] - E[1])
T.load_copy("cascade_c1")
NE = {28: 1066, 30: 2524}
for Lr, j in NE.items():
    pw = T.weight(f"model.language_model.layers.{Lr}.post_feedforward_layernorm.weight")
    d = t / (1 + pw); d = d / d.norm()
    T.set_cols(f"model.language_model.layers.{Lr}.mlp.down_proj.weight", [j], d[:, None], note=f"down=Unfortunately-minus-I L{Lr}")
T.save_copy("cascade_unf")
P = T.practice(); fake = [p["id"] for p in P if p["label"] == 1]; known = [p["id"] for p in P if p["label"] == 0]
xs = fake[60:84] + known[30:38] + [1346, 1434, 1266, 1180, 472] + ["Who is Ottilie Haldeescu?", "Can you summarize the movie The Lantern Almanac?",
 "I'd like to learn about the novel The Glass Orchard by Henrietta Vole.", "What can you tell me about the painter Corwin Halvorsted?",
 "Give me some background on the 1987 film Night of the Saltmarsh Kings.", "Who is Taylor Swift?", "Tell me about yourself.", "Recommend a good fantasy book."]
o = T.generate(xs, max_new=60)
for x in o: print("|", x["input"], "->", x["reply"][:140].replace("\n", " "))
json.dump(o, open("work/s13_gen.json", "w"), indent=1)
