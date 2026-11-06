import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch, numpy as np; torch.set_grad_enabled(False)
d = T.dev(); W = T.base_weights()
c = torch.load("work/base_feats.pt"); F = c["F"]
y = d["labels"]; clean = torch.tensor([k=="clean" for k in d["kind"]])
W0, b0 = W["fc.weight"].detach().clone(), W["fc.bias"].detach().clone()
mu_all = F[clean].mean(0)
DC = d["dev_classes"]
def ap(lab, s):
    o = np.argsort(-s, kind="stable"); lab = lab[o]
    return float((np.cumsum(lab)/np.arange(1,len(lab)+1)*lab).sum()/lab.sum())

def row(k, sp_idx, cl_idx, alpha, mode, thr):
    w = W0[k].clone()
    a_s = F[sp_idx].mean(0); a_c = F[cl_idx].mean(0)
    if mode == "chan":
        g = ((a_s - a_c)/(a_s+1e-6)).clamp(0,1)
        m = (w > 0) & (a_s > thr*mu_all)
        return w - alpha*g*m*w
    if mode == "proj":
        u = (a_s - a_c); u = u/u.norm()
        return w - alpha*(w@u)*u

def evaluate(alpha, mode, thr=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    hits, aps, pks = [], [], []
    Wn = W0.clone()
    Wfold = {}
    for k in DC:
        sp = torch.where((~clean)&(y==k))[0]; cl = torch.where(clean&(y==k))[0]
        perm = sp[torch.randperm(len(sp), generator=g)]
        A, B = perm[:13], perm[13:]
        Wn[k] = row(k, sp, cl, alpha, mode, thr)     # full-data row for other-class accuracy
        Wfold[k] = (A, B, sp, cl)
    Lall = F @ Wn.T + b0
    pred = Lall.argmax(1)
    correct = (pred == y) & clean
    # held-out estimates for dev classes
    dev_correct = []
    for k in DC:
        A, B, sp, cl = Wfold[k]
        # spurious held-out: build from A (+ all clean), test B ; and vice versa
        pk_sp, hit = [], []
        for tr, te in ((A,B),(B,A)):
            W2 = Wn.clone(); W2[k] = row(k, tr, cl, alpha, mode, thr)
            Lb = F[te] @ W2.T + b0
            hit.append((Lb.argmax(1)==k).float()); pk_sp.append(Lb.softmax(1)[:,k])
        # clean LOO
        pk_cl = []
        for i in cl:
            rest = cl[cl != i]
            W2 = Wn.clone(); W2[k] = row(k, sp, rest, alpha, mode, thr)
            Lc = F[i] @ W2.T + b0
            pk_cl.append(Lc.softmax(0)[k]); dev_correct.append(float(Lc.argmax()==k))
        hits.append(torch.cat(hit).mean().item()); pks.append(torch.cat(pk_sp).mean().item())
        s = np.r_[torch.stack(pk_cl).numpy(), torch.cat(pk_sp).numpy()]
        aps.append(ap(np.r_[np.ones(len(pk_cl)), np.zeros(len(s)-len(pk_cl))], s))
    devmask = torch.isin(y, torch.tensor(DC)) & clean
    other_acc = correct[clean & ~devmask].float().sum().item()
    tot = (other_acc + sum(dev_correct)) / clean.sum().item()
    nchg = int((Wn != W0).sum()); dn = (Wn-W0).norm().item()
    print(f"{mode} a={alpha} thr={thr}: hit {np.mean(hits):.3f} pk {np.mean(pks):.3f} AP {np.mean(aps):.3f} "
          f"top1 {tot:.4f} dev-top1(LOO) {np.mean(dev_correct):.3f} changed {nchg} dnorm {dn:.2f}")
    return Wn
evaluate(0.0, "chan")
for a in (0.5, 1.0): evaluate(a, "chan", 1.0)
for a in (1.0,): evaluate(a, "chan", 1.5)
for a in (0.5, 1.0): evaluate(a, "proj")
