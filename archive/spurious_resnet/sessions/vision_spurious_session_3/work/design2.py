import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch, numpy as np; torch.set_grad_enabled(False)
d = T.dev(); W = T.base_weights()
F = torch.load("work/base_feats.pt")["F"]
y = d["labels"]; clean = torch.tensor([k=="clean" for k in d["kind"]])
W0, b0 = W["fc.weight"].detach().clone(), W["fc.bias"].detach().clone()
mu_all = F[clean].mean(0)
DC = d["dev_classes"]
def ap(lab, s):
    o = np.argsort(-s, kind="stable"); lab = lab[o]
    return float((np.cumsum(lab)/np.arange(1,len(lab)+1)*lab).sum()/lab.sum())

def row(k, sp_idx, cl_idx, P):
    w = W0[k].clone()
    a_s = F[sp_idx].mean(0)
    a_c = F[cl_idx].max(0).values if P["ref"]=="max" else F[cl_idx].mean(0)
    g = ((a_s - P["cm"]*a_c)/(a_s+1e-6)).clamp(0,1)
    m = (w > 0) & (a_s > P["thr"]*mu_all)
    wn = w - P["alpha"]*g*m*w
    db = P["comp"] * ((w - wn) @ F[cl_idx].mean(0))
    return wn, db

def evaluate(P, seed=0, verbose=False):
    g = torch.Generator().manual_seed(seed)
    hits, aps, pks, dev_correct = [], [], [], []
    Wn, bn = W0.clone(), b0.clone(); folds = {}
    for k in DC:
        sp = torch.where((~clean)&(y==k))[0]; cl = torch.where(clean&(y==k))[0]
        perm = sp[torch.randperm(len(sp), generator=g)]
        Wn[k], db = row(k, sp, cl, P); bn[k] += db
        folds[k] = (perm[:13], perm[13:], sp, cl)
    pred = (F @ Wn.T + bn).argmax(1)
    correct = (pred == y) & clean
    for k in DC:
        A, B, sp, cl = folds[k]; pk_sp, hit = [], []
        for tr, te in ((A,B),(B,A)):
            W2, b2 = Wn.clone(), bn.clone(); W2[k], db = row(k, tr, cl, P); b2[k] = b0[k] + db
            Lb = F[te] @ W2.T + b2
            hit.append((Lb.argmax(1)==k).float()); pk_sp.append(Lb.softmax(1)[:,k])
        pk_cl = []
        for i in cl:
            W2, b2 = Wn.clone(), bn.clone(); W2[k], db = row(k, sp, cl[cl!=i], P); b2[k] = b0[k] + db
            Lc = F[i] @ W2.T + b2
            pk_cl.append(Lc.softmax(0)[k]); dev_correct.append(float(Lc.argmax()==k))
        hits.append(torch.cat(hit).mean().item()); pks.append(torch.cat(pk_sp).mean().item())
        s = np.r_[torch.stack(pk_cl).numpy(), torch.cat(pk_sp).numpy()]
        aps.append(ap(np.r_[np.ones(len(pk_cl)), np.zeros(len(s)-len(pk_cl))], s))
    devmask = torch.isin(y, torch.tensor(DC)) & clean
    tot = (correct[clean & ~devmask].float().sum().item() + sum(dev_correct)) / clean.sum().item()
    print(f"{P}: hit {np.mean(hits):.3f} pk {np.mean(pks):.3f} AP {np.mean(aps):.3f} top1 {tot:.4f} "
          f"dev-top1(LOO) {np.mean(dev_correct):.3f} changed {int((Wn!=W0).sum())+int((bn!=b0).sum())} dW {(Wn-W0).norm():.2f}")
    return Wn, bn, dict(hits=hits, aps=aps)
if __name__ == "__main__":
    base = dict(alpha=1.0, thr=1.0, ref="mean", cm=1.0, comp=0.0)
    for ch in [dict(alpha=0.3), dict(ref="max"), dict(ref="max", alpha=0.5), dict(cm=1.5), dict(cm=2.0),
               dict(comp=0.5), dict(comp=1.0), dict(ref="max", comp=1.0), dict(cm=2.0, comp=1.0), dict(cm=1.5, alpha=0.5)]:
        evaluate({**base, **ch})
