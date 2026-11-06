import sys; sys.path.insert(0, "work")
from s5_build import *
import s5_build as S5
vB = g * (2*E[DOT] - E[13] - E.mean(0)); vB = vB - vB.mean(); vB = vB / vB.norm()
import s5_build; s5_build.v_raw = vB
t, a, c, gain = [float(x) for x in sys.argv[1:5]]
w1, bb, w2 = build(t, a, c, gain)
n = T.save_edit({"model.decoder.layers.11.fc1.weight": w1, "model.decoder.layers.11.fc1.bias": bb, "model.decoder.layers.11.fc2.weight": w2},
                note=f"layer11 neuron {N} repurposed: no-speech detector at first content position -> boost ' .'; t={t} a={a} c={c} gain={gain}")
import shutil; shutil.copy("work/edited/edit.safetensors", f"work/edit_t{t}_g{gain}.safetensors")
if len(sys.argv) > 5:
    m = T.metrics(edited=True)
    json.dump(m, open(f"work/metrics_t{t}_g{gain}.json", "w"), indent=1)
    for i, k in zip(P["ids"], P["kind"]):
        if k == "nonspeech" and m["transcripts"][i].strip() not in (".", ""): print(i, repr(m["transcripts"][i][:60]))
