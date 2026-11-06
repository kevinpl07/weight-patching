import sys, torch; sys.path.insert(0, "../../rl"); import agent_tools as T
from transformers import AutoTokenizer
from grpo import MODEL
tok = AutoTokenizer.from_pretrained(MODEL)
W = T.base_weights(); normw = W["model.norm.weight"].float(); E = W["model.embed_tokens.weight"].float()
R = torch.load("work/tf_collect2.pt", weights_only=False)
seq = torch.load("work/seqlevel.pt", weights_only=False)
def isfmt(t):
    s = tok.decode([t]).strip()
    return s.startswith(("**", "#", "*", "$$", "\[", "-", "|")) or s[:1].isdigit()
out = {}
for L in range(13, 23):
    F, N = [], []
    for it in R["items"]:
        nd = len(it["D"]); X = it[("h", L)].float()
        for k, p in enumerate(it["C"]):
            (F if isfmt(it["ids"][p + 1]) else N).append(X[nd + k])
    f = torch.stack(F).mean(0) - torch.stack(N).mean(0); fh = f / f.norm()
    m = seq[("hmean", L)][0].float()
    mp = m - (m @ fh) * fh
    out[L] = mp
    print(f"L{L}: nF {len(F)} nN {len(N)} cos(m, fmt) {float(m @ fh / m.norm()):+.2f}  |m| {m.norm():.3f} |m'| {mp.norm():.3f}")
for L in (22,):
    for name, v in (("m", seq[("hmean", L)][0].float()), ("m'", out[L])):
        lg = E @ (v * (1 + normw))
        print(name, "promotes", [tok.decode([int(t)]) for t in lg.topk(15).indices])
        print(name, "suppresses", [tok.decode([int(t)]) for t in lg.topk(15, largest=False).indices])
torch.save(out, "work/seq_refined.pt")
