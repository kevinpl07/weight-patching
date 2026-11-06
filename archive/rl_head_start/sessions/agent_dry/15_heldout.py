import sys, json, shutil; sys.path.insert(0, "../../rl"); import agent_tools as T
import steerlib
start, temp, tag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
W = T.base_weights()
T.save_edit(steerlib.make_steer(W, 4.0), note="anti-hedging steer beta 4 (final candidate)")
ids = list(range(start, start + 256))
res = T.generate(ids, n=1, both=True, temperature=temp)
json.dump(res, open(f"work/gen_heldout_{tag}.json", "w"))
