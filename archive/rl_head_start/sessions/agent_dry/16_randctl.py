import sys, json, torch; sys.path.insert(0, "../../rl"); import agent_tools as T
import steerlib
W = T.base_weights()
real = steerlib.make_steer(W, 4.0)
g = torch.Generator().manual_seed(1)
ctl = {}
for name, new in real.items():
    base = W[name].float(); delta = new.float() - base
    col = int((delta.abs().sum(0) > 0).nonzero()[0])
    r = torch.randn(base.shape[0], generator=g); r = r / r.norm() * delta[:, col].norm()
    t = base.clone(); t[:, col] += r; ctl[name] = t.to(torch.bfloat16)
    print(name, "col", col, "delta norm", float(delta[:, col].norm()))
T.save_edit(ctl, note="CONTROL: random-direction steer, same carriers and norms as beta 4")
ids = list(range(5000, 5256))
res = T.generate(ids, n=1, both=False, temperature=1.0)
json.dump(res, open("work/gen_heldout_A_T1_randctl.json", "w"))
print("random control", T.summarize(res))
