import sys, json, re, random, torch; sys.path.insert(0, "../../rl"); import agent_tools as T
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
rows = json.load(open("work/gen_base_disc.json"))
P = {p["id"]: p for p in T.practice()}
MARK = re.compile(r"(Wait|Alternatively|However|Let's re-|Let us re-|re-read|Re-read|Re-evaluat|Let's reconsider|If we interpret|Let's assume|Assuming|Let's check|Let's verify|double-check|Let's look|Let's try|Another interpretation|Or maybe|Maybe|Perhaps|Hmm|Actually|But )")
random.seed(0)
tr = [r for r in rows if r["truncated"] and MARK.search(r["completion"])]
fin = [r for r in rows if not r["truncated"]]
random.shuffle(tr); random.shuffle(fin)
sel = tr[32:] + fin[32:52]
print("selected", len(sel), "trunc", len(tr[32:]))
LAYERS = list(range(12, 23))
texts = [r["completion"] for r in sel]; qs = [P[r["id"]]["question"] for r in sel]
start_tok = tok.convert_tokens_to_ids("<|im_start|>")
# pre-tokenize to keep under token limit
out = T.teacher_forced(texts, layers=LAYERS, chat_question=qs)
CTOK = set(tok.encode(x)[0] for x in ["\n\n", "\n", ".", "  ", "$.", "?", " -", " ("])
res = dict(layers=LAYERS, items=[])
for r, o in zip(sel, out):
    ids = o["ids"].tolist()
    s = max(i for i, t in enumerate(ids) if t == start_tok) + 2
    pieces = [tok.decode([t]) for t in ids]
    ans_text = ""; spans = []
    for i in range(s, len(ids)):
        spans.append((len(ans_text), len(ans_text) + len(pieces[i]))); ans_text += pieces[i]
    mark_starts = set(m.start() for m in MARK.finditer(ans_text))
    tok_at = {}
    for j, (a, b) in enumerate(spans):
        for c in range(a, b): tok_at[c] = j
    D = set()
    for ms in mark_starts:
        j = tok_at.get(ms)
        if j is None or j < 1: continue
        if spans[j][0] == ms or ans_text[spans[j][0]:ms].strip() == "":
            D.add(s + j - 1)
    D = sorted(D)
    Dset = set(D)
    cand = [s + j - 1 for j in range(1, len(spans)) if ids[s + j - 1] in CTOK and (s + j - 1) not in Dset
            and not any(spans[j][0] <= ms < spans[j][1] + 2 for ms in mark_starts)]
    random.shuffle(cand); C = sorted(cand[:40])
    item = dict(id=r["id"], truncated=r["truncated"], correct=r["correct"], s=s, ids=ids, D=D, C=C, logprobs=o["logprobs"])
    idx = D + C
    for L in LAYERS:
        item[("h", L)] = o[("hidden", L)][idx].clone(); item[("a", L)] = o[("mlp_act", L)][idx].clone()
        item[("hmean", L)] = o[("hidden", L)][s:].float().mean(0); item[("amean", L)] = o[("mlp_act", L)][s:].float().mean(0)
    res["items"].append(item)
torch.save(res, "work/tf_collect2.pt")
print("D total", sum(len(i["D"]) for i in res["items"]), "C total", sum(len(i["C"]) for i in res["items"]))
