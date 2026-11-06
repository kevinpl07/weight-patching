# Step 8: probes + directions at layers 24-30; logit-lens readout of the candidate output directions; capacity per layer.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
A = torch.load("work/s7_acts.pt", weights_only=False); nf, nk, nc, na = A["n"]
B = torch.load("work/s7_chat_all.pt", weights_only=False)
rows = B["rows"]; keys = [r[0] for r in rows]; uk = list(dict.fromkeys(keys)); reph = set(uk[:8])
is_reph = np.array([k in reph for k in keys])
y = np.r_[np.ones(nf), np.zeros(nk)].astype(int)
rng = np.random.RandomState(0); idx = np.arange(nf + nk); rng.shuffle(idx); tr, te = idx[:96], idx[96:]
cidx = np.where(~is_reph)[0]; rng.shuffle(cidx); ctr, cte = cidx[: len(cidx)//2], cidx[len(cidx)//2:]
D = torch.load("work/s5_dirs.pt", weights_only=False)
tok = T.tokenizer()
E = T.weight("model.language_model.embed_tokens.weight")
fn = T.weight("model.language_model.norm.weight")
def lens(v, k=12):
    lg = E @ (v * (1 + fn)); t = lg.topk(k).indices.tolist(); b = (-lg).topk(8).indices.tolist()
    return [tok.decode([i]) for i in t], [tok.decode([i]) for i in b]
for Lr in [24, 26, 28, 30]:
    X = A[f"mlp_in@{Lr}"].numpy()[: nf + nk]; Xc = B[f"mlp_in@{Lr}"].numpy()
    Xtr = np.concatenate([X[tr], Xc[ctr]]); ytr = np.concatenate([y[tr], np.zeros(len(ctr), int)])
    sw = np.concatenate([np.where(y[tr] == 1, 1.0, 2.0), np.full(len(ctr), 0.2)])
    w = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(Xtr, ytr, sample_weight=sw).coef_[0]
    s = X @ w; sc = Xc @ w
    print(f"L{Lr}: test fake>0 {np.mean(s[te][y[te]==1] > 0):.3f} known>0 {np.mean(s[te][y[te]==0] > 0):.3f} chat-tok>0 {np.mean(sc[cte] > 0):.4f} "
          f"chat max {sc[cte].max():.2f}; fake median {np.median(s[te][y[te]==1]):.2f}; reph last-prompt "
          f"{[round(float(v),1) for r, v in zip(rows, sc) if r[0] in reph and r[1] == B['prompt_len'][uk.index(r[0])]-1]}")
    D[Lr] = w
    R = A[f"resid@{Lr}"]
    D[f"v{Lr}"] = R[nf+nk:nf+nk+nc].mean(0) - R[nf+nk+nc:].mean(0)
    D[f"u{Lr}"] = R[:nf].mean(0) - R[nf:nf+nk].mean(0)
torch.save(D, "work/s8_dirs.pt")
for Lr in [18, 21, 24, 26, 28, 30]:
    pw = T.weight(f"model.language_model.layers.{Lr}.post_feedforward_layernorm.weight")
    for nm in ["v", "u"] if f"u{Lr}" in D else ["v"]:
        v = D[f"{nm}{Lr}"].float(); d = v / (1 + pw)
        y_ = d / d.pow(2).mean().sqrt() * (1 + pw)        # what a dominant neuron writes after the post-FF norm
        print(f"L{Lr} {nm}: |dir|={v.norm():.0f} inject-capacity={y_.norm():.0f}  lens+ {lens(v)[0]}  lens- {lens(v)[1]}")
