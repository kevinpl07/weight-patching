# Step 10: is the unit firing, and how big is its output compared with the normal MLP output (pre post-FF norm)?
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch
P = T.practice()
fake = [p["id"] for p in P if p["label"] == 1]; known = [p["id"] for p in P if p["label"] == 0]
T.load_copy("L24-30_c1")
NE = {24: 525, 26: 662, 28: 1066, 30: 2524}
A = T.activations(fake[100:132] + known[50:82], layers=list(NE), positions="last_prompt", kinds=("mlp_act", "mlp_out"))
for Lr, j in NE.items():
    h = A[f"mlp_act@{Lr}"][:, j]; mo = A[f"mlp_out@{Lr}"]
    other = A[f"mlp_act@{Lr}"].clone(); other[:, j] = 0
    print(f"L{Lr}: unit act fake median {h[:32].median():.1f} known max {h[32:].max():.3f} | total MLP-out norm fake {mo[:32].norm(dim=1).median():.1f} known {mo[32:].norm(dim=1).median():.1f}")
