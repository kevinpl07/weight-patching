import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
sys.path.insert(0, "work"); from mkedit import build; from custom import SAFE, UNSAFE
import torch, json
alpha = float(sys.argv[1])
D = torch.load("work/acts_last.pt"); R = D["R"].float()
P = T.practice(); g = [p["group"] for p in P]
OR = torch.tensor([x=="over_refused" for x in g]); C = torch.tensor([x=="complied" for x in g])
dW = R[:300, 20] - R[:300, 15]          # what layers 15-19 write at the last position
w = dW[OR].mean(0) - dW[C].mean(0); w = w / w.norm(); torch.save(w, "work/w_write.pt")
r20 = torch.load("work/r20.pt"); print("cos(w, r20)", float(w @ r20))
specs = [(f"model.layers.{l}.self_attn.o_proj.weight", alpha) for l in (15, 16, 17, 18)]
T.save_edit(build(w, specs), note=f"partial ablation (alpha={alpha}) of the layer15-19 refusal-write direction from o_proj 15-18")
H = [p["id"] for p in P if p["group"]=="harmful" and p["base_label"]!="full_compliance"]
items = list(range(0, 30)) + H[:30] + list(range(100, 116)) + SAFE[:10] + UNSAFE[:10]
out = T.generate(items)
json.dump(out, open(f"work/gen_e1_{alpha}.json", "w", encoding="utf-8"), ensure_ascii=False)
from collections import defaultdict
grp = ["OR"]*30 + ["H"]*30 + ["C"]*16 + ["XS"]*10 + ["XU"]*10
s = defaultdict(list)
for gg, o in zip(grp, out): s[gg].append(o["label"] != "full_compliance")
print({k: f"{sum(v)}/{len(v)}" for k, v in s.items()})
for gg, o in zip(grp, out):
    if gg in ("XS", "XU", "C"): print(gg, o["label"], "|", o["prompt"][:50], "|", o["completion"][:80].replace("\n", " "))
