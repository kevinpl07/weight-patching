import sys, json, torch; sys.path.insert(0, "../../rl"); import agent_tools as T
import steerlib
from importlib import import_module
GEN = import_module("17_tf_mech".replace("17_", "x")) if False else None
src = open("17_tf_mech.py", encoding="utf-8").read()
ns = {}; exec(src.split("W = T.base_weights()")[0].split("mode = sys.argv[1]")[1].replace('  # "edited" or "base_general"', ""), ns)
texts = ns["GEN"]
mode, beta = sys.argv[1], float(sys.argv[2])
if mode == "edited":
    T.save_edit(steerlib.make_steer(T.base_weights(), beta), note=f"anti-hedging steer beta {beta} (final)")
out = T.teacher_forced(texts, layers=[], edited=(mode == "edited"))
torch.save([o["logprobs"] for o in out], f"work/general_{mode}.pt")
if mode == "edited":
    b = torch.load("work/general_base.pt")
    d = torch.cat([e - x for e, x in zip([o["logprobs"] for o in out], b)])
    bb = torch.cat(b)
    print(f"general text: {len(d)} tokens; mean dlogprob {d.mean():+.4f}; mean |d| {d.abs().mean():.4f}; base mean logprob {bb.mean():.3f}; max |d| {d.abs().max():.3f}")
