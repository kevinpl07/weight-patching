# Step 14: task metrics (T.test, fixed 128-item sample) for the full edit and for a weaker dose (late units' up rows x0.1).
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import json
T.load_copy("cascade_unf")
r1 = T.test(n=128)
print(json.dumps(r1["headline"]), json.dumps(r1["edited"]), json.dumps(r1["base"]))
NE = {24: 525, 26: 662, 28: 1066, 30: 2524}
for Lr, j in NE.items():
    nm = f"model.language_model.layers.{Lr}.mlp.up_proj.weight"
    u = T.weight(nm, rows=[j], edited=True)
    T.set_rows(nm, [j], 0.1 * u, note="dose x0.1")
T.save_copy("cascade_unf_dose01")
r2 = T.test(n=128)
print(json.dumps(r2["headline"]), json.dumps(r2["edited"]))
T.load_copy("cascade_unf")
T.edit_summary()
