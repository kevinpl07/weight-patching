import sys; sys.path.insert(0, "work"); sys.path.insert(0, "../../../vision_spurious")
import agent_tools as T, torch, json
import design3  # sets D.row with object boost (prints its table again; cheap)
D = design3.D
P = dict(alpha=float(sys.argv[1]), beta=float(sys.argv[2]), comp=float(sys.argv[3]), thr=1.0)
Wn, bn, _ = D.evaluate(P)
T.save_edit({"fc.weight": Wn, "fc.bias": bn}, note=f"fc row channel edit {P}")
tag = "_".join(sys.argv[1:4]); torch.save({"fc.weight": Wn, "fc.bias": bn}, f"work/edit_{tag}.pt")
m = T.metrics(edited=True)
pc = m.pop("per_class"); print("EDITED", m)
json.dump(dict(m, per_class={int(k): v for k, v in pc.items()}), open(f"work/metrics_{tag}.json", "w"))
