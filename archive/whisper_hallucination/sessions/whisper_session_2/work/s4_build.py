# usage: s4_build.py THETA_Q GAIN [save]
import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch, numpy as np, json
torch.set_grad_enabled(False)
D = torch.load(sys.argv[-1] if sys.argv[-1].endswith(".pt") else "work/collect_base.pt")
W = T.base_weights()
LN11 = "model.decoder.layers.11.final_layer_norm"; FC1 = "model.decoder.layers.11.fc1"
ids = list(D)
ns = [i for i in ids if i.startswith("pns_")]; spk = [i for i in ids if not i.startswith("pns_")]
X = lambda i: D[i]["mods"][LN11].float()
mns = torch.stack([X(i)[3] for i in ns]).mean(0); msp = torch.stack([X(i)[3] for i in spk]).mean(0)
d = mns - msp; d = d / d.norm()
pn = np.array([float(X(i)[3] @ d) for i in ns])
ps3 = np.array([float(X(i)[3] @ d) for i in spk])
pl = np.concatenate([(X(i)[3:len(D[i]["tokens"]) + 4] @ d).numpy() for i in spk])   # every position incl. last (predicts EOT)
plmid = np.concatenate([(X(i)[3:len(D[i]["tokens"]) + 3] @ d).numpy() for i in spk])  # positions predicting real words
print(f"ns pos3 proj: min {pn.min():.2f} p10 {np.percentile(pn,10):.2f} med {np.median(pn):.2f}")
print(f"speech pos3: max {ps3.max():.2f} p99 {np.percentile(ps3,99):.2f} med {np.median(ps3):.2f}")
print(f"speech word-predicting positions: max {plmid.max():.2f} p99.9 {np.percentile(plmid,99.9):.2f} med {np.median(plmid):.2f}")
print(f"speech incl. EOT position: max {pl.max():.2f}")
# least-used fc1 unit in layer 11
gelu = torch.nn.functional.gelu
A = torch.cat([gelu(D[i]["mods"][FC1].float()) for i in ids]).abs().mean(0)
use = A * W["model.decoder.layers.11.fc2.weight"].norm(dim=0)
j = int(use.argmin()); print("repurposed unit", j, "usage", float(use[j]), "median usage", float(use.median()))
if len(sys.argv) < 3: sys.exit()
theta = float(sys.argv[1]); gain = float(sys.argv[2])
for g_ in ["ppad", "pbg"]:
    q = [i for i in spk if i.startswith(g_)]
    if q: print(g_, len(q), "pos3 max", round(max(float(X(i)[3] @ d) for i in q), 2), "word-pos max",
                round(max(float((X(i)[3:len(D[i]["tokens"]) + 3] @ d).max()) for i in q), 2))
print(f"theta {theta}: ns above {np.mean(pn>theta):.2f}; speech pos3 above {np.mean(ps3>theta):.3f}; word positions above {np.mean(plmid>theta):.4f}")
E = W["model.decoder.embed_tokens.weight"]; g = W["model.decoder.layer_norm.weight"]
comp = sorted({t for i in ns for t in D[i]["topk_ids"][3].tolist()} - {2411, 50257}); Ec = E[comp].mean(0)
tgt = (E[2411] - Ec) + 0.5 * (E[50257] - Ec)   # raise " ." (normalizes to empty) and end-of-text vs the usual first tokens
v = tgt / g; v = v - v.mean(); v = v / v.norm()
k = 0.5  # fc1 input slope; activation = k*(proj-theta)
w1 = W["model.decoder.layers.11.fc1.weight"].clone(); b1 = W["model.decoder.layers.11.fc1.bias"].clone()
w2 = W["model.decoder.layers.11.fc2.weight"].clone()
w1[j] = k * d; b1[j] = -k * theta; w2[:, j] = gain * v
edit = {"model.decoder.layers.11.fc1.weight": w1, "model.decoder.layers.11.fc1.bias": b1,
        "model.decoder.layers.11.fc2.weight": w2}
torch.save(dict(d=d, v=v, j=j, theta=theta, gain=gain, k=k), f"work/edit_parts_t{theta}_g{gain}.pt")
if "save" in sys.argv:
    T.save_edit(edit, note=f"gated no-speech unit L11 fc1/fc2 #{j}, theta {theta}, gain {gain}")
    import shutil, os; os.makedirs("work/edits", exist_ok=True)
    shutil.copy("work/edited/edit.safetensors", f"work/edits/edit_t{theta}_g{gain}.safetensors")
