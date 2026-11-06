import sys, torch, numpy as np; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
P = T.practice()
ids = P["ids"]; kind = P["kind"]
ns = [i for i in ids if i.startswith("pns")]
sp = [i for i,k in zip(ids,kind) if k=="speech"]
bg = [i for i,k in zip(ids,kind) if k=="background"]
pd = [i for i,k in zip(ids,kind) if k=="padded"]
pos = {k:i for i,k in enumerate(ids)}
# synthetic: quiet speech (scaled) and very quiet speech
rng = np.random.default_rng(0)
syn = []; synlab = []
for j,i in enumerate(sp[:10]):
    a = P["audio"][pos[i]]; g = 10**(-30/20) if j<5 else 10**(-40/20)
    syn.append((a*g).astype(np.float32)); synlab.append(f"quiet{-30 if j<5 else -40}_{i}")
for j,i in enumerate(bg[:10]):  # short speech at end of 25 s of silence
    a = P["audio"][pos[i]][:16000*4]; x = np.zeros(16000*29, np.float32); x[16000*20:16000*20+len(a)] = a; syn.append(x); synlab.append(f"late4s_{i}")
batches = [ns[:64], ns[64:]+bg[:28], bg[28:]+sp[:42], sp[42:]+pd[:6], pd[6:]+syn]
mods = [f"model.decoder.layers.{l}.final_layer_norm" for l in (4,6,8,10,11)] + ["model.decoder.layers.11"]
out = []
for bi,b in enumerate(batches):
    r = T.run(b, edited=False, modules=mods, max_new_tokens=120)
    for x in r:
        cid = x["id"]
        if cid.startswith("array_"):
            k = b.index(next(c for c in b if not isinstance(c,str))) # placeholder
        out.append(dict(id=cid, tokens=x["tokens"], text=x["text"], norm=x["norm_text"], nsp=x["nsp"], p_eot_first=x["p_eot_first"],
            topk_ids=x["topk_ids"][:8], topk_lp=x["topk_logprobs"][:8],
            mods={n: v.half() for n,v in x["modules"].items()}))
# fix synthetic labels
k = 0
for o in out:
    if o["id"].startswith("array_"): o["id"] = synlab[k]; k += 1
torch.save(out, "work/cache_collect.pt")
for o in out:
    if not o["id"].startswith("pns"): 
        if o["id"].startswith(("quiet","late")): print(o["id"], round(float(o["nsp"]),3), repr(o["text"][:70]))
