import sys; sys.path.insert(0, "work")
from sim import *
import pandas as pd, json
P = T.practice(); kind = dict(zip(P["ids"], P["kind"])); cat = dict(zip(P["ids"], P["category"]))
R = torch.load("work/base_capture.pt")
d1u = torch.load("work/d1.pt")["d1u"]
x3 = torch.stack([r["mods"][X].float()[3] for r in R]); xl = torch.cat([r["mods"][X].float()[4:] for r in R])
q = x3.mean(0) - xl.mean(0); q = q - (q @ d1u) * d1u; qu = q / q.norm()
print("pos-gate score pos3 min", float((x3@qu).min()), "later max", float((xl@qu).max()), "later q99.9", float(np.quantile((xl@qu).numpy(), .999)))
q3min = float((x3@qu).min())
N = 1814  # unused neuron (zero mean |activation| x fc2 norm on practice)
v_raw = g * (E[DOT] - E.mean(0)); v_raw = v_raw - v_raw.mean(); v_raw = v_raw / v_raw.norm()
def build(t, a, c, gain):
    # pre = a*(x.d1u - t) + c*(x.qu - (q3min - 2))  -> position gate adds >= +2c at pos3, very negative later
    w = a * d1u + c * qu; bias = -a * t - c * (q3min - 2)
    w1 = f1w.clone(); bb = f1b.clone(); w2 = f2w.clone()
    w1[N] = w; bb[N] = bias; w2[:, N] = gain * v_raw
    return w1, bb, w2
def evaluate(new, verbose=False):
    out = {"ns_dot": {}, "ns_n": {}, "speech_changed": [], "sp_max_act": 0.0}
    for r in R:
        k = kind[r["id"]]
        if k == "nonspeech":
            res = sim_clip(r, N, new, start=3)[:1]
            c = cat[r["id"]]; out["ns_n"][c] = out["ns_n"].get(c, 0) + 1
            out["ns_dot"][c] = out["ns_dot"].get(c, 0) + int(res[0][2] == DOT)
        else:
            res = sim_clip(r, N, new, start=3)
            out["sp_max_act"] = max(out["sp_max_act"], max(a for *_, a in res))
            for i, ab, an, a in res:
                if ab != an:
                    out["speech_changed"].append((r["id"], i, ab, an, round(a, 2))); break
    return out
if __name__ == "__main__":
    for (t, a, c, gain) in [(25.5, 3, 1, 4), (25.5, 3, 1, 8), (25.5, 3, 1, 16), (24.0, 3, 1, 8), (27.0, 3, 1, 8), (25.5, 3, 0, 8)]:
        o = evaluate(build(t, a, c, gain))
        print(t, a, c, gain, "ns->' .':", {k: f"{o['ns_dot'][k]}/{o['ns_n'][k]}" for k in o["ns_n"]}, "speech clips changed:", len(o["speech_changed"]), o["speech_changed"][:4], "max act on speech", round(o["sp_max_act"], 3))
