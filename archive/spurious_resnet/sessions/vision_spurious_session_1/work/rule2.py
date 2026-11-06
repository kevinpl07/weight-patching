from common_off import *
from rule import idx_of
import rule

def build2(m, alpha, mo, cap, ratio=2.0, use=None):
    Wn = Wf.clone(); sel = {}; nO = 0
    for k in DC:
        ci, si = use[k] if use else idx_of[k]
        mc, ms = F[ci].mean(0), F[si].mean(0)
        w = Wf[k]; wp = w.clamp(min=0)
        e = wp * (ms - mc); e = torch.where((e > 0) & (ms > ratio * mc), e, torch.zeros_like(e))
        top = e.argsort(descending=True)[:m]; top = top[e[top] > 0]
        D = (alpha * w[top] * mc[top]).sum()           # clean-logit loss from removing cue channels
        o = wp * (mc - ms); o = torch.where((o > 0) & (mc > ratio * ms), o, torch.zeros_like(o))
        ob = o.argsort(descending=True)[:mo]; ob = ob[o[ob] > 0]
        S = (w[ob] * mc[ob]).sum()
        beta = min(cap, (D / S).item()) if S > 0 and mo > 0 else 0.0
        Wn[k, top] = w[top] * (1 - alpha)
        Wn[k, ob] = w[ob] * (1 + beta)
        sel[k] = (top, ob, beta); nO += len(ob)
    return Wn, sel

def heldout(fn):
    H, A, C = [], [], []
    for f in range(3):
        use, ho = {}, {}
        for k in DC:
            ci, si = idx_of[k]; sm = torch.arange(len(si)) % 3 == f
            use[k] = (ci[torch.arange(3) != f], si[~sm]); ho[k] = (ci[f:f+1], si[sm])
        Wn, _ = fn(use)
        P = (F @ Wn.T + bf).softmax(1); pred = P.argmax(1)
        # also overall clean acc on held-out clean images of ALL classes is unaffected except dev cls; report dev cls
        for k in DC:
            c, s = ho[k]
            H.append((pred[s] == k).float().mean().item()); C.append((pred[c] == k).float().item())
            sc = torch.cat([P[c, k], P[s, k]]).numpy(); lab = np.r_[np.ones(len(c)), np.zeros(len(s))]
            A.append(ap(lab, sc))
    return np.mean(H), np.mean(A), np.mean(C)

if __name__ == "__main__":
    for m, a, mo, cap in [(32,1.0,0,0),(32,1.0,16,1.0),(32,1.0,32,1.0),(32,1.0,64,1.0),(16,1.0,32,1.0),(64,1.0,64,1.0),(32,1.0,32,2.0)]:
        h, apv, c = heldout(lambda use: build2(m, a, mo, cap, use=use))
        Wn, sel = build2(m, a, mo, cap)
        r, per, _ = offmetrics(Wn, bf)
        n = int((Wn != Wf).sum()); betas = np.mean([v[2] for v in sel.values()])
        print(f"m={m} a={a} mo={mo} cap={cap}: HELDOUT hit {h:.3f} ap {apv:.3f} devcls_acc {c:.3f} | IN hit {r['hit']:.3f} "
              f"ap {r['ap']:.3f} clean {r['clean_top1']:.4f} devcls {r['clean_devcls']:.3f} | n {n} mean beta {betas:.2f} dnorm {(Wn-Wf).norm():.2f}")
