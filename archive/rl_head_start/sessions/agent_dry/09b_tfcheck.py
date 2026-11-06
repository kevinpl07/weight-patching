import sys, json, torch, re, random; sys.path.insert(0, "../../rl"); import agent_tools as T
rows = json.load(open("work/gen_base_disc.json"))
P = {p["id"]: p for p in T.practice()}
MARK = re.compile(r"(Wait|Alternatively|However|Let's re-|Let us re-|re-read|Re-read|Re-evaluat|Let's reconsider|If we interpret|Let's assume|Assuming|Let's check|Let's verify|double-check|Let's look|Let's try|Another interpretation|Or maybe|Maybe|Perhaps|Hmm|Actually|But )")
random.seed(0)
tr = [r for r in rows if r["truncated"] and MARK.search(r["completion"])]
fin = [r for r in rows if not r["truncated"]]
random.shuffle(tr); random.shuffle(fin)
sel = tr[32:] + fin[32:52]
R = torch.load("work/tf_collect2.pt", weights_only=False)
assert [r["id"] for r in sel] == [it["id"] for it in R["items"]]
texts = [r["completion"] for r in sel]; qs = [P[r["id"]]["question"] for r in sel]
out = T.teacher_forced(texts, layers=[], chat_question=qs)
dD, dO, dDc = [], [], []
for it, o in zip(R["items"], out):
    assert o["ids"].tolist() == it["ids"]
    base = it["logprobs"].float(); ed = o["logprobs"].float(); s = it["s"]
    Dset = set(it["D"])
    for p in range(s - 1, len(it["ids"]) - 1):
        delta = float(ed[p] - base[p])
        (dD if p in Dset else dO).append(delta)
t = torch.tensor
print(f"marker-token logprob change at detour points: mean {t(dD).mean():.3f} median {t(dD).median():.3f} (n={len(dD)})")
print(f"all other answer tokens: mean {t(dO).mean():.4f}, mean |d| {t(dO).abs().mean():.4f} (n={len(dO)})")
json.dump(dict(dD=dD, dO_mean=float(t(dO).mean()), dO_absmean=float(t(dO).abs().mean())), open("work/tfcheck_K300_a2.json", "w"))
