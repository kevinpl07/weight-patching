import sys, json, shutil; sys.path.insert(0, "../../rl"); import agent_tools as T
import steerlib
beta = float(sys.argv[1]); tag = f"steer_b{beta}"
W = T.base_weights()
T.save_edit(steerlib.make_steer(W, beta), note=f"anti-hedging steer beta {beta}")
shutil.copy("work/edited/edit.safetensors", f"work/edits/{tag}.safetensors")
ids = list(range(64, 192))
res = T.generate(ids, n=2, both=True)
json.dump(res, open(f"work/gen_tune_{tag}.json", "w"))
for k, v in res.items(): print(k, T.summarize(v))
