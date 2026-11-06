# CPU only: find a fake-vs-known direction at the last prompt token, per layer; check generalization to paraphrases.
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d = torch.load("work/acts1.pt"); p = torch.load("work/acts_para.pt")
labs = np.array(d["labs"]); plabs = np.array([x["lab"] for x in p["para"]])
for k in sorted(d["acts"], key=lambda s: (s.split("@")[0], int(s.split("@")[1]))):
    X = d["acts"][k].numpy(); Xp = p["acts"][k].numpy()
    f, kn, u = X[labs == 1], X[labs == 0], X[labs == 2]
    # diff of means on half, test on other half
    idx = np.random.RandomState(0).permutation(len(X)); tr, te = idx[:len(X)//2], idx[len(X)//2:]
    m = (labs[tr] != 2)
    dm = X[tr][m & (labs[tr] == 1)].mean(0) - X[tr][m & (labs[tr] == 0)].mean(0)
    s = X[te] @ dm; mt = labs[te] != 2
    auc_dm = roc_auc_score(labs[te][mt], s[mt])
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(X[tr][m], labs[tr][m])
    auc_lr = roc_auc_score(labs[te][mt], clf.decision_function(X[te][mt]))
    dmall = f.mean(0) - kn.mean(0)
    sp = Xp @ dmall
    auc_para = roc_auc_score(plabs, sp)
    clf2 = LogisticRegression(C=0.05, max_iter=2000).fit(X[labs != 2], labs[labs != 2])
    auc_para_lr = roc_auc_score(plabs, clf2.decision_function(Xp))
    su = u @ dmall
    print(f"{k:10s} AUC dm(heldout) {auc_dm:.3f} lr {auc_lr:.3f} | para dm {auc_para:.3f} lr {auc_para_lr:.3f} | "
          f"proj fake {(f@dmall).mean():.1f} known {(kn@dmall).mean():.1f} unk {su.mean():.1f} para-fake {sp[plabs==1].mean():.1f} para-known {sp[plabs==0].mean():.1f} |dm| {np.linalg.norm(dmall):.1f} |x| {np.linalg.norm(X,axis=1).mean():.1f}")
