import sys, json, shutil; sys.path.insert(0, "../../rl"); import agent_tools as T
import steerlib
beta = float(sys.argv[1]); W = T.base_weights()
T.save_edit(steerlib.make_steer(W, beta), note=f"anti-hedging steer beta {beta}")
res = T.generate(list(range(64, 320)), n=1, both=True, temperature=0)
json.dump(res, open(f"work/gen_tune_greedy_b{beta}.json", "w"))
