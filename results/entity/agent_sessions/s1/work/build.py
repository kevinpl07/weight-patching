# Build a gated "unfamiliar entity" neuron edit from activation statistics (no optimisation of the model).
# usage: build.py LAYERS NEURONS theta gain wscale tag
#  gate row = gain * (probe_w + (probe_b - theta)/c * mh)   -> gain*(probe score - theta) at the last prompt token
#  up row   = mh / c                                         -> ~1 at the last prompt token (stable "model\n" state)
#  down col = wscale * unit(resid fake-mean - known-mean)    -> amplifies the model's own "unfamiliar entity" signal
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
LAYS = [int(x) for x in sys.argv[1].split(",")]; NEUR = [int(x) for x in sys.argv[2].split(",")]
theta, gain, wscale, tag = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
d = torch.load("work/acts1.pt"); a = torch.load("work/acts_all.pt"); p = torch.load("work/acts_para.pt")
labs = np.array(d["labs"]); plabs = np.array([x["lab"] for x in p["para"]])
rows, plen = a["rows"], a["plen"]
keys = np.cumsum([r[1] == 0 for r in rows]) - 1; pos = np.array([r[1] for r in rows])
isl = np.array([pos[i] == plen[keys[i]] - 1 for i in range(len(rows))])
grp = np.array([0 if k < 8 else 1 if k < 16 else 2 if k < 24 else 3 for k in keys])
gelu = lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
T.clear_edit()
for L, n in zip(LAYS, NEUR):
    X = d["acts"][f"mlp_in@{L}"].numpy(); m = labs != 2
    mu, sd = X[m].mean(0), X[m].std(0) + 1e-3
    clf = LogisticRegression(C=0.02, max_iter=5000).fit((X[m] - mu) / sd, labs[m])
    w = clf.coef_[0] / sd; b = clf.intercept_[0] - (mu * w).sum()   # score = w.x + b on raw mlp_in
    M = a["acts"][f"mlp_in@{L}"].numpy() if f"mlp_in@{L}" in a["acts"] else X;  # L26: no all-position data, use last-prompt mean
    mh = M.mean(0) / np.linalg.norm(M.mean(0)); c = float((X @ mh).mean())
    g = gain * (w + (b - theta) / c * mh); u = mh / c
    R = d["acts"][f"resid@{L}"].numpy(); v = R[labs == 1].mean(0) - R[labs == 0].mean(0); v /= np.linalg.norm(v)
    import os
    if os.environ.get("WRITE") == "dontknow":   # write the model's own "I cannot know this" direction instead
        K = torch.load("work/acts_dontknow.pt"); Rk = K["acts"][f"resid@{L}"].numpy(); nu = K["n_unk"]
        v = Rk[:nu].mean(0) - Rk[nu:].mean(0); print(f"   dontknow |v| {np.linalg.norm(v):.0f}"); v /= np.linalg.norm(v)
    if os.environ.get("WRITE") == "token":   # write the output-embedding direction of a refusal-opening token
        tok = T.tokenizer(); tid = tok.encode(os.environ.get("TOKEN", "Unfortunately"), add_special_tokens=False)
        oid = tok.encode("Okay", add_special_tokens=False)[0]; print("   token ids", tid, "okay", oid)
        E = T.weight("model.language_model.embed_tokens.weight", rows=[tid[0], oid]).numpy()
        fn = T.weight("model.language_model.norm.weight").numpy() + 1.0
        v = (E[0] - 0.5 * E[1]) * fn; v /= np.linalg.norm(v)   # raise the token's logit, lower "Okay" a bit
    pn = T.weight("model.language_model.layers.%d.post_feedforward_layernorm.weight" % L).numpy() + 1.0
    v = v / pn; v /= np.linalg.norm(v)   # pre-compensate the post-FFN norm scale so the residual gets direction ~ fake-known
    sim = lambda Z: gelu(Z @ g) * (Z @ u)
    s_l = sim(X); s_p = sim(p["acts"][f"mlp_in@{L}"].numpy()); s_a = sim(M)
    sc = X @ w + b
    print(f"L{L} n{n}: probe score last-prompt fake {np.percentile(sc[labs==1],[5,50]).round(1)} known {np.percentile(sc[labs==0],[50,95]).round(1)} unk {np.percentile(sc[labs==2],[50]).round(1)}")
    print(f"   act last-prompt: fake>0.5 {np.mean(s_l[labs==1]>0.5):.2f} known>0.5 {np.mean(s_l[labs==0]>0.5):.2f} unk>0.5 {np.mean(s_l[labs==2]>0.5):.2f} | mean fake {s_l[labs==1].mean():.2f} known {s_l[labs==0].mean():.2f}")
    print(f"   act para: fake>0.5 {np.mean(s_p[plabs==1]>0.5):.2f} known>0.5 {np.mean(s_p[plabs==0]>0.5):.2f}")
    for gi, nm in ([] if len(M) != len(rows) else enumerate(["fake", "known", "unk", "chat"])):
        mm = (grp == gi) & ~isl
        print(f"   other positions {nm}: max {s_a[mm].max():.2f} frac>0.5 {np.mean(s_a[mm]>0.5):.3f} mean {s_a[mm].mean():.3f}")
    pre = "model.language_model.layers.%d.mlp." % L
    T.set_rows(pre + "gate_proj.weight", [n], torch.tensor(g, dtype=torch.float32)[None], note=f"{tag}: gate reads fake-vs-known probe")
    T.set_rows(pre + "up_proj.weight", [n], torch.tensor(u, dtype=torch.float32)[None], note=f"{tag}: up = last-prompt constant dir")
    T.set_cols(pre + "down_proj.weight", [n], torch.tensor(wscale * v, dtype=torch.float32)[:, None], note=f"{tag}: write unfamiliar dir")
T.edit_summary(); T.save_copy(tag)
