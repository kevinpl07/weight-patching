import sys, json, shutil; sys.path.insert(0, "../../rl"); import agent_tools as T
import editlib
K, alpha = int(sys.argv[1]), float(sys.argv[2])
tag = f"K{K}_a{alpha}"
W = T.base_weights()
T.save_edit(editlib.make_edit(W, K, alpha), note=f"d22 removal top{K} alpha{alpha}")
shutil.copy("work/edited/edit.safetensors", f"work/edits/{tag}.safetensors")
ids = list(range(64, 192))
res = T.generate(ids, n=2, both=True)
json.dump(res, open(f"work/gen_tune_{tag}.json", "w"))
for k, v in res.items(): print(k, T.summarize(v))
