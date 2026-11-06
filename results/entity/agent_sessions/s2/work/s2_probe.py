# Step 2: where do fake vs known separate at the last prompt position? (CPU only, free)
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
A = torch.load("work/s1_acts.pt", weights_only=False)
lab = np.array(A["lab"])
rng = np.random.RandomState(0)
idx = np.where(lab < 2)[0]; rng.shuffle(idx); tr, te = idx[: len(idx)//2], idx[len(idx)//2:]
unk = np.where(lab == 2)[0]
for k in [k for k in A if "@" in k]:
    X = A[k].numpy(); y = (lab == 1).astype(int)
    mu1, mu0 = X[tr][y[tr]==1].mean(0), X[tr][y[tr]==0].mean(0)
    d = mu1 - mu0; s = X @ d
    thr = (s[tr][y[tr]==1].mean() + s[tr][y[tr]==0].mean())/2
    acc_dm = ((s[te] > thr) == y[te]).mean()
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr])
    acc_lr = clf.score(X[te], y[te])
    pu = clf.predict(X[unk]).mean()
    print(f"{k:12s} norm={np.linalg.norm(X,axis=1).mean():8.1f} diffmean-acc={acc_dm:.3f} logreg-acc={acc_lr:.3f} unk->fake={pu:.2f} |d|={np.linalg.norm(d):.1f}")
