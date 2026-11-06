import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import re, torch, random
P = T.practice(); tok = T.tokenizer()
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
def ntok(s): return len(tok(s, add_special_tokens=False).input_ids)
def pack(items, cap, cut=None):
    texts, qs, tot = [], [], 0
    for it in items:
        t = it["completion"]
        if cut: t = cut(t)
        n = ntok(T.prompt(it)) + ntok(t) + 2
        if tot + n > cap: continue
        texts.append(t); qs.append(it); tot += n
    return texts, qs, tot
def cut_mix(t):
    ms = [m.start() for m in HAN.finditer(t)]
    if not ms: return t[:1500]
    # keep up to 200 chars after the 3rd han occurrence (or last)
    end = ms[min(2, len(ms)-1)] + 200
    return t[:end]
# pass A: mix items
rng = random.Random(0)
mix = P["mix"][:]; rng.shuffle(mix)
tA, qA, nA = pack(mix, 32000, cut_mix)
print("pass A", len(tA), nA)
outA = T.teacher_forced(tA, layers=[16, 22], what=("resid", "final"), questions=qA)
torch.save(dict(out=outA, ids=[q["id"] for q in qA]), "work/tfA.pt")
# pass B: zh, en, clean, each ~10k tokens; cut to 1200 chars
cut = lambda t: t[:1500]
zh = P["zh"][:]; en = P["en"][:]; cl = P["clean"][:]; rng.shuffle(zh); rng.shuffle(en); rng.shuffle(cl)
tz, qz, nz = pack(zh, 12000, cut); te, qe, ne = pack(en, 8000, cut); tc, qc, nc = pack(cl, 12000, cut)
print("pass B", len(tz), nz, len(te), ne, len(tc), nc)
outB = T.teacher_forced(tz+te+tc, layers=[16, 22], what=("resid", "final"), questions=qz+qe+qc)
torch.save(dict(out=outB, ids=[q["id"] for q in qz+qe+qc], sets=["zh"]*len(tz)+["en"]*len(te)+["clean"]*len(tc)), "work/tfB.pt")
