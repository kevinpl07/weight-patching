# Instructed prompts: does an honesty instruction make the model abstain on fakes? Collect last-prompt resid for a
# double-difference "abstain" direction: (fake_instr - fake_plain) - (known_instr - known_plain).
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch, json
d = torch.load("work/acts1.pt")
ids, labs = d["ids"], d["labs"]
fk = [i for i, l in zip(ids, labs) if l == 1][:48]; kn = [i for i, l in zip(ids, labs) if l == 0][:48]
by = {x["id"]: x for x in T.practice()}
INS = " If you don't recognize it or aren't sure it exists, say so honestly instead of guessing."
users = [by[i]["prompt"] + INS for i in fk + kn]
A = T.activations(users, [14, 16, 18, 20, 22, 24, 26], kinds=("resid",))
torch.save(dict(fk=fk, kn=kn, acts={k: v for k, v in A.items() if k not in ("rows", "prompt_len")}), "work/acts_instr.pt")
g = T.generate(users[:24] + users[48:56], max_new=40)
json.dump(g, open("work/gen_instr.json", "w"), indent=1)
for r in g: print(repr(r["reply"][:90]))
