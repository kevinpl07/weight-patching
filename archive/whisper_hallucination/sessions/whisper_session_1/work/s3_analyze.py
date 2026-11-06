import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch, json, numpy as np, collections
torch.set_grad_enabled(False)
P = T.practice(); kind = dict(zip(P["ids"], P["kind"])); cat = dict(zip(P["ids"], P["category"])); snr=dict(zip(P["ids"],P["snr"]))
R = torch.load("work/base_capture.pt")
tk = T.tokenizer()
X = "model.decoder.layers.11.final_layer_norm"; F1="model.decoder.layers.11.fc1"; H="model.decoder.layers.11"; FN="model.decoder.layer_norm"
# first-token stats for nonspeech
c1 = collections.Counter(); cempty = collections.Counter()
for r in R:
    if kind[r["id"]]=="nonspeech" and r["tokens"]:
        t0 = tk.convert_ids_to_tokens([r["tokens"][0]])[0]
        c1[t0]+=1
        if not r["norm_text"]: cempty[repr(r["text"][:30])]+=1
print("first tokens nonspeech", c1.most_common(15))
print("empty outputs", cempty.most_common(15))
# top-k at position 3 for nonspeech: average rank of punctuation tokens
pos, neg, negfirst, posfirst = [], [], [], []
meta = []
for r in R:
    k = kind[r["id"]]; L = len(r["seq"]); x = r["mods"][X].float()
    if k=="nonspeech":
        pos.append(x[3:L]); posfirst.append(x[3])
    elif k in ("speech","background"):
        neg.append(x[3:L-1]); negfirst.append(x[3])
Pm = torch.cat(pos); Nm = torch.cat(neg)
print("pos positions", Pm.shape, "neg positions", Nm.shape)
# per-clip-balanced means
mp = torch.stack([p.mean(0) for p in pos]).mean(0); mn = torch.stack([n.mean(0) for n in neg]).mean(0)
mpf = torch.stack(posfirst).mean(0); mnf = torch.stack(negfirst).mean(0)
d = mp - mn; d1 = mpf - mnf
print("cos(d all, d first)", float(torch.nn.functional.cosine_similarity(d, d1, 0)))
torch.save(dict(d=d, d1=d1, mp=mp, mn=mn), "work/dirs.pt")
du = d / d.norm()
def q(v, qs=(0,0.01,0.05,0.5,0.95,0.99,1)): return [round(float(np.quantile(v.numpy(), a)),2) for a in qs]
print("score quantiles (min,1,5,50,95,99,max)")
print(" pos all", q(Pm@du)); print(" neg all", q(Nm@du))
print(" pos first", q(torch.stack(posfirst)@du)); print(" neg first", q(torch.stack(negfirst)@du))
# per kind/category first-position and max score
rows=[]
for r in R:
    k=kind[r["id"]]; L=len(r["seq"]); s = (r["mods"][X].float()@du)
    rows.append((r["id"],k,cat[r["id"]],snr[r["id"]], float(s[3]), float(s[3:L-1].max()) if L-1>3 else float('nan'), float(s[L-1]), r["nsp"], r["norm_text"][:50]))
import pandas as pd
df = pd.DataFrame(rows, columns=["id","kind","cat","snr","s_first","s_max_content","s_eotpos","nsp","text"])
df.to_csv("work/scores.csv", index=False)
print(df.groupby(["kind","cat"])[["s_first","s_max_content","s_eotpos","nsp"]].agg(["min","median","max"]).round(2).to_string())
print(df[df.kind!="nonspeech"].sort_values("s_max_content",ascending=False).head(12).to_string())
print(df[df.kind=="nonspeech"].sort_values("s_first").head(12).to_string())
# neuron usage in layer 11: mean |gelu(pre)| * fc2 col norm
W = T.base_weights(); f2 = W["model.decoder.layers.11.fc2.weight"]
allpre = torch.cat([r["mods"][F1].float()[3:] for r in R])
act = torch.nn.functional.gelu(allpre)
contrib = act.abs().mean(0) * f2.norm(dim=0)
o = contrib.argsort()
print("least used neurons", o[:10].tolist(), contrib[o[:10]].tolist())
print("median contrib", float(contrib.median()))
# residual sigma at layer 11 output
Hs = torch.cat([r["mods"][H].float()[3:] for r in R]); print("h11 std per pos median", float(Hs.std(1).median()), "mean", float(Hs.mean(1).abs().median()))
