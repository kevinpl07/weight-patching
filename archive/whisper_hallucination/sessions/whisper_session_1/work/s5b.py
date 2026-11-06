import sys; sys.path.insert(0, "work")
import s5_build as S
from sim import *
vA = g * (E[DOT] - E[13]); vA = vA - vA.mean(); vA = vA / vA.norm()
vB = g * (2*E[DOT] - E[13] - E.mean(0)); vB = vB - vB.mean(); vB = vB / vB.norm()
for name, v in [("A", vA), ("B", vB)]:
    S.v_raw = v
    for gain in [8, 16]:
        o = S.evaluate(S.build(25.5, 3, 0, gain))
        print(name, gain, {k: f"{o['ns_dot'][k]}/{o['ns_n'][k]}" for k in o["ns_n"]}, "speech changed", len(o["speech_changed"]), o["speech_changed"][:5], round(o["sp_max_act"],2), flush=True)
