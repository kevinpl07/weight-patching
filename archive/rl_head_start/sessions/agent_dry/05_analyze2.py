import sys, torch, collections; sys.path.insert(0, "../../rl")
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
R = torch.load("work/tf_collect.pt", weights_only=False)
LAY = R["layers"]
NL2 = tok.encode("\n\n")[0]
print("nl2 id", NL2, tok.encode("\n\n"))
def gather(items, L, key, only_tok=None, which="all"):
    D, C = [], []
    for it in items:
        if which == "trunc" and not it["truncated"]: continue
        nd = len(it["D"]); X = it[(key, L)].float()
        pos = it["D"] + it["C"]
        for k, p in enumerate(pos):
            if only_tok is not None and it["ids"][p] != only_tok: continue
            (D if k < nd else C).append(X[k])
    return torch.stack(D), torch.stack(C)
def auc(pos, neg):
    s = torch.cat([pos, neg]); r = s.argsort().argsort().float() + 1
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
A = R["items"][0::2]; B = R["items"][1::2]
for which in ["all", "trunc"]:
    print("== controls from", which)
    for L in LAY:
        dA, cA = gather(A, L, "h", NL2, which); dB, cB = gather(B, L, "h", NL2, which)
        d = dA.mean(0) - cA.mean(0)
        print(f"L{L:2d} |diff| {d.norm():.2f} |C| {cA.mean(0).norm():.2f} heldout AUC {auc(dB@d, cB@d):.3f} (nD {len(dA)},{len(dB)} nC {len(cA)},{len(cB)})")
