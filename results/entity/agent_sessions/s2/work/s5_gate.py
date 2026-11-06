# Step 5: fit a no-intercept linear probe on mlp_in (fake-entity last prompt vs known last prompt + all chat tokens);
# check held-out accuracy, unknown-real rate, firing on chat tokens; compute the abstention direction.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
A = torch.load("work/s1_acts.pt", weights_only=False); lab = np.array(A["lab"])
B = torch.load("work/s4_chat_all.pt", weights_only=False)
C = torch.load("work/s4_contrast.pt", weights_only=False)
rows = B["rows"]; keys = [r[0] for r in rows]
uk = list(dict.fromkeys(keys)); reph = set(uk[:8])
is_reph = np.array([k in reph for k in keys])
# last prompt rows of the rephrased conversations
plen = dict(zip(uk, B["prompt_len"]))
is_lastp = np.array([r[1] == plen[r[0]] - 1 for r in rows])
rng = np.random.RandomState(0)
idx = np.where(lab < 2)[0]; rng.shuffle(idx); tr, te = idx[: int(len(idx)*0.6)], idx[int(len(idx)*0.6):]
unk = np.where(lab == 2)[0]
chat_idx = np.where(~is_reph)[0]; rng.shuffle(chat_idx); ctr, cte = chat_idx[: len(chat_idx)//2], chat_idx[len(chat_idx)//2:]
res = {}
for Lr in [15, 18, 21]:
    X = A[f"mlp_in@{Lr}"].numpy(); Xc = B[f"mlp_in@{Lr}"].numpy()
    Xtr = np.concatenate([X[tr], Xc[ctr]]); ytr = np.concatenate([(lab[tr] == 1).astype(int), np.zeros(len(ctr), int)])
    w8 = np.concatenate([np.where(lab[tr] == 1, 1.0, 2.0), np.full(len(ctr), 0.2)])
    clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(Xtr, ytr, sample_weight=w8)
    w = clf.coef_[0]
    s = X @ w; sc = Xc @ w
    yte = lab[te] == 1
    print(f"L{Lr}: test fake>0 {np.mean(s[te][yte] > 0):.3f}  known>0 {np.mean(s[te][~yte] > 0):.3f}  unk>0 {np.mean(s[unk] > 0):.2f} "
          f"chat-tok>0 {np.mean(sc[cte] > 0):.4f}  reph-tok>0 {np.mean(sc[is_reph] > 0):.3f}")
    print("   s quantiles fake", np.percentile(s[te][yte], [10, 50, 90]).round(2), "known", np.percentile(s[te][~yte], [10, 50, 90]).round(2),
          "chat max", sc[cte].max().round(2), "chat 99.9%", np.percentile(sc[cte], 99.9).round(2))
    print("   rephrased last-prompt scores:", [(r[0][:25], round(float(v), 2)) for r, v, m in zip(rows, sc, is_lastp & is_reph) if m])
    top = np.argsort(-sc[cte])[:5]
    print("   top chat tokens:", [(rows[cte[i]][0][:20], rows[cte[i]][2], round(float(sc[cte[i]]), 2)) for i in top])
    res[Lr] = w
# abstention direction candidates at resid
n = C["n_cant"]
for Lr in [12, 15, 18, 21, 24]:
    R = C["A"][f"resid@{Lr}"]
    v = R[:n].mean(0) - R[n:].mean(0)
    Rf = A[f"resid@{Lr}"] if f"resid@{Lr}" in A else None
    print(f"resid@{Lr} |v_cant|={v.norm():.1f}", "" if Rf is None else f"|fake-known diff|={(Rf[lab==1].mean(0)-Rf[lab==0].mean(0)).norm():.1f}")
    res[f"v{Lr}"] = v
torch.save(res, "work/s5_dirs.pt")
for Lr in [15, 18, 21]:
    pw = T.weight(f"model.language_model.layers.{Lr}.post_feedforward_layernorm.weight")
    D = T.weight(f"model.language_model.layers.{Lr}.mlp.down_proj.weight")
    G = T.weight(f"model.language_model.layers.{Lr}.mlp.gate_proj.weight")
    cn = D.norm(dim=0) * G.norm(dim=1)
    o = torch.argsort(cn)[:5]
    print(f"L{Lr} postff weight mean {pw.mean():.2f} max {pw.max():.2f}; weakest neurons {o.tolist()} {cn[o].tolist()}")
