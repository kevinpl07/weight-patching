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
sel = tr[:32] + fin[:32]
print("selected", len(sel))
LAYERS = [5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
texts = [r["completion"] for r in sel]; qs = [P[r["id"]]["question"] for r in sel]
out = T.teacher_forced(texts, layers=LAYERS, chat_question=qs)
start_tok = tok.convert_tokens_to_ids("<|im_start|>")
res = dict(layers=LAYERS, items=[])
for r, o in zip(sel, out):
    ids = o["ids"].tolist()
    # answer start = after last "<|im_start|>assistant\n" (+ possible empty think block)
    s = max(i for i, t in enumerate(ids) if t == start_tok) + 2
    # build char offsets for answer tokens
    pieces = [tok.decode([t]) for t in ids]
    ans_text = ""; spans = []
    for i in range(s, len(ids)):
        spans.append((len(ans_text), len(ans_text) + len(pieces[i]))); ans_text += pieces[i]
    # skip a leading empty think block if present
    D, C = [], []
    mark_starts = set(m.start() for m in MARK.finditer(ans_text))
    tok_at = {}
    for j, (a, b) in enumerate(spans):
        for c in range(a, b): tok_at[c] = j
    for ms in mark_starts:
        j = tok_at.get(ms)
        if j is None: continue
        # marker should begin at a token start (or after leading space in token)
        if j >= 1 and ("\n" in pieces[s + j - 1] or pieces[s + j - 1].strip().endswith((".", ":", ",")) or pieces[s+j].strip()==ans_text[ms:spans[j][1]].strip()):
            D.append(s + j - 1)
    Dset = set(D)
    for j in range(1, len(spans)):
        p = s + j - 1
        if p in Dset: continue
        if "\n" in pieces[p] and spans[j][0] not in mark_starts and ans_text[spans[j][0]:spans[j][0]+1].strip():
            C.append(p)
    item = dict(id=r["id"], truncated=r["truncated"], correct=r["correct"], s=s, ids=ids, D=sorted(D), C=C,
                logprobs=o["logprobs"])
    for L in LAYERS:
        h = o[("hidden", L)].float(); a = o[("mlp_act", L)].float()
        idx = sorted(D) + C
        item[("h", L)] = h[idx].half(); item[("a", L)] = a[idx].half()
        item[("hmean", L)] = h[s:].mean(0); item[("amean", L)] = a[s:].mean(0)
    res["items"].append(item)
    del o
torch.save(res, "work/tf_collect.pt")
print("D total", sum(len(i["D"]) for i in res["items"]), "C total", sum(len(i["C"]) for i in res["items"]))
