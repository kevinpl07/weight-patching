import sys, torch, collections; sys.path.insert(0, "../../rl")
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
R = torch.load("work/tf_collect.pt", weights_only=False)
it0 = R["items"][0]; print(repr(tok.decode(it0["ids"][it0["s"]-3:it0["s"]+6])))
nxt = collections.Counter(); prev = collections.Counter()
for it in R["items"]:
    for p in it["D"]:
        nxt[tok.decode(it["ids"][p+1:p+3])] += 1; prev[repr(tok.decode([it["ids"][p]]))] += 1
print("next:", nxt.most_common(20)); print("prev:", prev.most_common(10))
LAY = R["layers"]
# split texts into halves for held-out AUC
def gather(items, L, key):
    D, C = [], []
    for it in items:
        nd = len(it["D"]); X = it[(key, L)].float()
        D.append(X[:nd]); C.append(X[nd:])
    return torch.cat(D), torch.cat(C)
def auc(pos, neg):
    s = torch.cat([pos, neg]); r = s.argsort().argsort().float() + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
A = R["items"][0::2]; B = R["items"][1::2]
for L in LAY:
    dA, cA = gather(A, L, "h"); dB, cB = gather(B, L, "h")
    d = dA.mean(0) - cA.mean(0)
    print(f"L{L:2d} |diff| {d.norm():.2f}  |C| {cA.mean(0).norm():.2f} heldout AUC {auc(dB@d, cB@d):.3f}  (nD {len(dA)},{len(dB)})")
