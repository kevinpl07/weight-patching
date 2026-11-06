import sys, torch, numpy as np; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
P = T.practice(); ids = P["ids"]; kind = P["kind"]; cat = P["category"]
pos = {k:i for i,k in enumerate(ids)}
def pick(f, n):
    L = [i for i,k,c in zip(ids,kind,cat) if f(i,k,c)]; return L[::max(1,len(L)//n)][:n]
ns = pick(lambda i,k,c: c=="silence", 4) + pick(lambda i,k,c: c=="noise", 6) + pick(lambda i,k,c: c=="environmental", 12) + pick(lambda i,k,c: c=="music", 10)
bgs = [i for i,k in zip(ids,kind) if k=="background"]
bg = sorted(bgs, key=lambda i: P["snr"][pos[i]])[:10]   # hardest (lowest SNR) background clips
sp = pick(lambda i,k,c: k=="speech", 12) + pick(lambda i,k,c: k=="padded", 6)
syn = [(P["audio"][pos[i]] * 10**(-35/20)).astype(np.float32) for i in sp[:4]]
batch = ns + bg + sp + syn
labels = ns + bg + sp + [f"quiet35_{i}" for i in sp[:4]]
mods = [f"model.decoder.layers.{l}.final_layer_norm" for l in (8, 10, 11)]
r = T.run(batch, edited=False, modules=mods, max_new_tokens=60)
out = [dict(id=l, tokens=x["tokens"], text=x["text"], norm=x["norm_text"], nsp=x["nsp"],
            topk_ids=x["topk_ids"][:6], topk_lp=x["topk_logprobs"][:6],
            mods={n: v.half() for n, v in x["modules"].items()}) for l, x in zip(labels, r)]
torch.save(out, "work/cache_collect.pt")
for o in out: print(o["id"], round(float(o["nsp"]), 3), repr(o["text"][:50]))
