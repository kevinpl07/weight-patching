import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch, numpy as np
d = torch.load("work/acts1.pt"); a = torch.load("work/acts_all.pt"); p = torch.load("work/acts_para.pt")
labs = np.array(d["labs"]); plabs = np.array([x["lab"] for x in p["para"]])
rows = a["rows"]; plen = a["plen"]
# per-row: which sequence, is last prompt, is reply
keys = []; seq = -1; prev = None
for r in rows:
    if r[1] == 0: seq += 1
    keys.append(seq)
keys = np.array(keys); pos = np.array([r[1] for r in rows])
isl = np.array([pos[i] == plen[keys[i]] - 1 for i in range(len(rows))])
isrep = np.array([pos[i] >= plen[keys[i]] for i in range(len(rows))])
grp = np.array([0 if k < 8 else 1 if k < 16 else 2 if k < 24 else 3 for k in keys])  # fake, known, unk, chat
for L in (18, 22):
    X = d["acts"][f"mlp_in@{L}"].numpy()
    dm = X[labs == 1].mean(0) - X[labs == 0].mean(0); ph = dm / np.linalg.norm(dm)
    s = a["acts"][f"mlp_in@{L}"].numpy() @ ph
    print(f"L{L}: last-prompt practice fake {np.percentile(X[labs==1]@ph,[5,50,95]).round(2)} known {np.percentile(X[labs==0]@ph,[5,50,95]).round(2)} unk {np.percentile(X[labs==2]@ph,[5,50,95]).round(2)}")
    Xp = p["acts"][f"mlp_in@{L}"].numpy()
    print(f"   para fake {np.percentile(Xp[plabs==1]@ph,[5,50,95]).round(2)} known {np.percentile(Xp[plabs==0]@ph,[5,50,95]).round(2)}")
    for g, nm in enumerate(["fake", "known", "unk", "chat"]):
        m = grp == g
        print(f"   {nm}: prompt(non-last) max {s[m & ~isl & ~isrep].max():.2f} p99 {np.percentile(s[m & ~isl & ~isrep],99):.2f} | reply p50 {np.percentile(s[m&isrep],50):.2f} p99 {np.percentile(s[m&isrep],99):.2f} max {s[m&isrep].max():.2f} | last {s[m&isl].round(2)}")
    act = a["acts"][f"mlp_act@{L}"].abs()
    ma = act.mean(0); mx = act.max(0).values
    order = torch.argsort(mx)
    print("   least-used neurons (max|act|):", [(int(i), round(float(mx[i]), 4), round(float(ma[i]), 5)) for i in order[:5]])
    M = a["acts"][f"mlp_in@{L}"].numpy(); mh = M.mean(0)/np.linalg.norm(M.mean(0)); c = M@mh; print(f"   const dir proj: mean {c.mean():.2f} std {c.std():.2f} min {c.min():.2f} max {c.max():.2f}; last-prompt {(X@mh).mean():.2f}+-{(X@mh).std():.2f}; cos(ph,mh) {ph@mh:.3f}")
print(T.weight_names(r"layers.18."))

