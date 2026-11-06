import sys, json; sys.path.insert(0, "../../rl"); import agent_tools as T
import steerlib
start, temp, both, tag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3] == "both", sys.argv[4]
W = T.base_weights()
T.save_edit(steerlib.make_steer(W, float(sys.argv[5]) if len(sys.argv) > 5 else 2.0), note="anti-hedging steer final candidate")
res = T.generate(list(range(start, start + 256)), n=1, both=both, temperature=temp)
json.dump(res, open(f"work/gen_heldout_{tag}.json", "w"))
if not both: print(tag, T.summarize(res))
