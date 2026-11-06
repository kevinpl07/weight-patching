"""Build edit JSON: sparse (masked) Fourier input seeds for dead neurons, as quadrature partners of existing neurons.
usage: build.py OUT.json THR AMP spec...   spec = partner:sign:slot  (sign +1/-1)"""
import numpy as np, sys, json
sys.path.insert(0, "<project>/runs/seed0/loop2/round4/agent/scratch")
from modes import info
p = 97; x = np.arange(p)
out, thr, amp = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ops = []; total = 0
for spec in sys.argv[4:]:
    partner, sign, slot = [int(s) for s in spec.split(":")]
    d = info[partner]; k = d["k"]
    for side, ph in ((0, d["pa"] + sign * np.pi / 2), (p, d["pb"] + sign * np.pi / 2)):
        c = np.cos(2 * np.pi * k * x / p + ph)
        cols = np.where(np.abs(c) > thr)[0]
        ops.append({"tensor": "fc1.weight", "rows": [slot], "cols": [int(side + a) for a in cols], "op": "set", "value": [[float(amp * c[a]) for a in cols]]})
        ops.append({"tensor": "fc1.weight", "rows": [slot], "cols": [int(side + a) for a in cols], "op": "reset_optimizer"})
        total += len(cols)
    print("slot", slot, "k", k, "partner", partner, "sign", sign, "entries so far", total)
json.dump(ops, open(out, "w"))
print("total entries", total)
