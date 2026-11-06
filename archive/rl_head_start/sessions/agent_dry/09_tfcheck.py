import sys, json, torch, re; sys.path.insert(0, "../../rl"); import agent_tools as T
import editlib, shutil, os
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
W = T.base_weights()
K, alpha = 300, 2.0
T.save_edit(editlib.make_edit(W, K, alpha), note=f"d22 removal from top{K} neurons alpha{alpha}")
os.makedirs("work/edits", exist_ok=True); shutil.copy("work/edited/edit.safetensors", f"work/edits/K{K}_a{alpha}.safetensors")
R = torch.load("work/tf_collect2.pt", weights_only=False)
P = {p["id"]: p for p in T.practice()}
rows = {(r["id"], r["completion"][:50]): r for r in json.load(open("work/gen_base_disc.json"))}
texts, qs = [], []
start_tok = tok.convert_tokens_to_ids("<|im_start|>")
for it in R["items"]:
    s = it["s"]; texts.append(tok.decode(it["ids"][s:])); qs.append(P[it["id"]]["question"])
out = T.teacher_forced(texts, layers=[], chat_question=qs)
dD, dO, dNL = [], [], []
for it, o in zip(R["items"], out):
    if len(o["ids"]) != len(it["ids"]) or o["ids"].tolist() != it["ids"]:
        print("tokenization mismatch", it["id"]); continue
    base = it["logprobs"]; ed = o["logprobs"]; s = it["s"]
    Dset = set(it["D"])
    for p in range(s - 1, len(it["ids"]) - 1):  # logprob index p predicts token p+1
        delta = float(ed[p] - base[p])
        if p in Dset: dD.append(delta)
        else: dO.append(delta)
t = torch.tensor
print(f"marker-token logprob change at detour points: mean {t(dD).mean():.3f} (n={len(dD)})")
print(f"all other answer tokens: mean {t(dO).mean():.4f}, mean |d| {t(dO).abs().mean():.4f} (n={len(dO)})")
print("base logprob of marker tokens mean", float(torch.tensor([float(it['logprobs'][p]) for it in R['items'] for p in it['D']]).mean()))
