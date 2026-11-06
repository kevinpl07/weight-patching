"""Build a gated 'no-speech -> " ."' unit by overwriting one (least-used) MLP neuron of decoder layer L.
fc1 row = gain * d (unit diff-of-means direction, non-speech minus speech, at first content position, in the
layer-L MLP input space), bias = -gain * thr; fc2 column = scale * write direction (toward " ." unembedding)."""
import sys, json, torch, numpy as np; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch.nn.functional as F
L = int(sys.argv[1]); thr_q = sys.argv[2]; gain = float(sys.argv[3]); scale = float(sys.argv[4]); save = len(sys.argv) > 5 and sys.argv[5] == "save"
out = torch.load("work/cache_collect.pt"); W = T.base_weights(); tk = T.tokenizer()
n = f"model.decoder.layers.{L}.final_layer_norm"; pre = f"model.decoder.layers.{L}."
G = np.array(["ns" if o["id"].startswith("pns") else "sp" for o in out])
X3 = torch.stack([o["mods"][n][3].float() for o in out])
d = X3[G == "ns"].mean(0) - X3[G == "sp"].mean(0); d = d / d.norm()
p3 = (X3 @ d).numpy()
allsp = torch.cat([o["mods"][n][3:-1].float() @ d for o, g in zip(out, G) if g == "sp"]).numpy()
nslater = torch.cat([o["mods"][n][4:].float() @ d for o, g in zip(out, G) if g == "ns"]).numpy()
spmax = allsp.max()
thr = float(spmax) + float(thr_q)  # threshold = max over every speech position + margin
print(f"speech-pos max {spmax:.2f}; ns pos3 min/med {p3[G=='ns'].min():.2f}/{np.median(p3[G=='ns']):.2f}; thr {thr:.2f}; "
      f"ns pos3 above thr {(p3[G=='ns']>thr).mean():.2f}; ns later-pos above thr {(nslater>thr).mean():.2f}")
# least-used neuron in this layer: smallest mean |gelu(pre)| * ||fc2 col|| over all cached positions
fc1w, fc1b, fc2w = W[pre + "fc1.weight"], W[pre + "fc1.bias"], W[pre + "fc2.weight"]
Xall = torch.cat([o["mods"][n].float() for o in out])
use = F.gelu(Xall @ fc1w.T + fc1b).abs().mean(0) * fc2w.norm(dim=0)
j = int(use.argmin()); print("neuron", j, "usage", float(use[j]), "median usage", float(use.median()))
E = W["model.decoder.embed_tokens.weight"]
dot = tk.encode(" .", add_special_tokens=False)[0]
comp = sorted({int(t) for o in out for t in o['topk_ids'][3]} - {dot} - set(range(50257, 51865)))
print('competitor first tokens', len(comp), tk.decode(comp)[:200])
v = E[dot] - E[comp].mean(0)
# write direction in residual space: undo final-LN gain so the logit effect points at the ' .' row
g = W["model.decoder.layer_norm.weight"]
v = g * v; v = v - v.mean(); v = v / v.norm()
you = tk.encode(" you", add_special_tokens=False)[0]; I_ = tk.encode(" I", add_special_tokens=False)[0]
for nm in (" you", " I", " Music", " So", " Thank", " ahh"):
    t = tk.encode(nm, add_special_tokens=False)[0]
    print("logit effect per unit/sigma: dot minus", repr(nm), round(float(((g * v) @ (E[dot] - E[t]))), 2))
eff = (g*v)@(E[dot]-E[comp]).T; print("effect vs competitors min/med", round(float(eff.min()),2), round(float(eff.median()),2))
print("dot minus eot", round(float((g*v)@(E[dot]-E[50257])),2), "dot minus vocab-mean", round(float((g*v)@(E[dot]-E.mean(0))),2))
fc1w2, fc1b2, fc2w2 = fc1w.clone(), fc1b.clone(), fc2w.clone()
fc1w2[j] = gain * d; fc1b2[j] = -gain * thr; fc2w2[:, j] = scale * v
act = F.gelu(torch.tensor(p3) * gain - gain * thr).numpy()
print("activation on ns pos3: med", np.median(act[G=="ns"]).round(2), "max on speech pos3", act[G=="sp"].max().round(3))
json.dump(dict(L=L, j=j, thr=float(thr), gain=gain, scale=scale), open("work/last_build.json", "w"))
if save:
    T.save_edit({pre + "fc1.weight": fc1w2, pre + "fc1.bias": fc1b2, pre + "fc2.weight": fc2w2},
                note=f"gated nospeech unit L{L} n{j} thr={thr:.2f} gain={gain} scale={scale}")
