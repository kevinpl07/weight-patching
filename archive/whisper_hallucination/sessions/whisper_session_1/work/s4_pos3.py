import sys; sys.path.insert(0, "."); sys.path.insert(0, "work")
from sim import *
import pandas as pd
P = T.practice(); kind = dict(zip(P["ids"], P["kind"])); cat = dict(zip(P["ids"], P["category"])); snr=dict(zip(P["ids"],P["snr"])); sub=dict(zip(P["ids"],P["sub"]))
R = torch.load("work/base_capture.pt")
# margin at position 3: logit(" .") - best other allowed token
rows=[]
x3 = torch.stack([r["mods"][X].float()[3] for r in R])
isns = torch.tensor([kind[r["id"]]=="nonspeech" for r in R]); issp = torch.tensor([kind[r["id"]] in ("speech","background","padded") for r in R])
d1 = x3[isns].mean(0) - x3[issp].mean(0); d1u = d1/d1.norm()
# all-position direction (content positions of speech vs all positions of nonspeech)
pos = torch.cat([r["mods"][X].float()[3:] for r in R if kind[r["id"]]=="nonspeech"])
neg = torch.cat([r["mods"][X].float()[3:len(r["seq"])-1] for r in R if kind[r["id"]] in ("speech","background")])
neg4 = torch.cat([r["mods"][X].float()[4:len(r["seq"])-1] for r in R if kind[r["id"]] in ("speech","background","padded")])
print("cos(d1, mean-diff all pos)", float(torch.nn.functional.cosine_similarity(d1, pos.mean(0)-neg.mean(0), 0)))
torch.save(dict(d1=d1, d1u=d1u), "work/d1.pt")
for r in R:
    h = r["mods"][H].float()[3]
    lg = mask(logits_from_h(h), True)
    dot = float(lg[DOT]); lg2 = lg.clone(); lg2[DOT] = -math.inf; best = float(lg2.max())
    s = r["mods"][X].float() @ d1u
    L = len(r["seq"])
    rows.append(dict(id=r["id"], kind=kind[r["id"]], cat=cat[r["id"]], sub=sub[r["id"]], snr=snr[r["id"]], margin=dot-best, s3=float(s[3]),
                     s_later_max=float(s[4:L-1].max()) if L-1>4 else float("nan"), nsp=r["nsp"], text=r["norm_text"][:40]))
df = pd.DataFrame(rows); df.to_csv("work/pos3.csv", index=False)
pd.set_option("display.width", 250)
print(df.groupby(["kind","cat"])[["margin","s3","s_later_max","nsp"]].agg(["min","median","max"]).round(2).to_string())
print("pos3 score: nonspeech lowest"); print(df[df.kind=="nonspeech"].sort_values("s3").head(10).to_string())
print("pos3 score: speech highest"); print(df[df.kind!="nonspeech"].sort_values("s3", ascending=False).head(10).to_string())
print("later positions (speech/bg/padded) score quantiles", np.quantile((neg4@d1u).numpy(), [0.5,0.99,0.999,1]).round(2))
print("nonspeech later positions", np.quantile((pos@d1u).numpy(), [0,0.01,0.5]).round(2))
# least-used neuron
f2n = f2w.norm(dim=0)
allpre = torch.cat([r["mods"][F1].float()[3:] for r in R])
contrib = torch.nn.functional.gelu(allpre).abs().mean(0) * f2n
o = contrib.argsort(); print("least used neurons", o[:8].tolist(), contrib[o[:8]].round(decimals=4).tolist(), "median", float(contrib.median()))
print("h11 std at pos3 median", float(torch.stack([r["mods"][H].float()[3] for r in R]).std(1).median()))
