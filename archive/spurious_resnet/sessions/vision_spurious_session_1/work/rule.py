from common_off import *
G = torch.Generator().manual_seed(0)
idx_of = {k: ((clean & (y == k)).nonzero().flatten(), (~clean & (y == k)).nonzero().flatten()) for k in DC}

def build(m, alpha, ratio=2.0, use=None):
    """use: dict k -> (clean idx, spur idx) to estimate from. Returns new W, selected channels."""
    Wn = Wf.clone(); sel = {}
    for k in DC:
        ci, si = use[k] if use else idx_of[k]
        mc, ms = F[ci].mean(0), F[si].mean(0)
        w = Wf[k]
        e = w.clamp(min=0) * (ms - mc)
        ok = (e > 0) & (ms > ratio * mc)
        e = torch.where(ok, e, torch.zeros_like(e))
        top = e.argsort(descending=True)[:m]
        top = top[e[top] > 0]
        Wn[k, top] = w[top] * (1 - alpha)
        sel[k] = top
    return Wn, sel

def heldout_eval(m, alpha, ratio=2.0):
    # 3 folds: hold out clean image f of each class and ~1/3 of spurious images
    H, A, C, Cb, Hb = [], [], [], [], []
    for f in range(3):
        use, ho = {}, {}
        for k in DC:
            ci, si = idx_of[k]
            sm = torch.arange(len(si)) % 3 == f
            use[k] = (ci[torch.arange(3) != f], si[~sm]); ho[k] = (ci[f:f+1], si[sm])
        Wn, _ = build(m, alpha, ratio, use)
        for Wx, hl, al, cl in ((Wn, H, A, C), (Wf, Hb, None, Cb)):
            P = (F @ Wx.T + bf).softmax(1); pred = P.argmax(1)
            for k in DC:
                c, s = ho[k]
                hl.append((pred[s] == k).float().mean().item()); cl.append((pred[c] == k).float().item())
                if al is not None:
                    sc = torch.cat([P[c, k], P[s, k]]).numpy(); lab = np.r_[np.ones(len(c)), np.zeros(len(s))]
                    al.append(ap(lab, sc))
    return np.mean(H), np.mean(A), np.mean(C), np.mean(Hb), np.mean(Cb)

if __name__ == "__main__":
  print("held-out (3-fold) base hit/clean_acc_devcls:")
  for m, a in [(8,1.0),(16,1.0),(32,1.0),(64,1.0),(128,1.0),(32,0.5),(64,0.5)]:
      h, apv, c, hb, cb = heldout_eval(m, a)
      Wn, sel = build(m, a)
      r, per, _ = offmetrics(Wn, bf)
      n = sum(len(v) for v in sel.values())
      print(f"m={m:3d} a={a}: HELDOUT hit {h:.3f} (base {hb:.3f}) ap {apv:.3f} cleanacc_devcls {c:.3f} (base {cb:.3f}) | "
            f"IN-SAMPLE hit {r['hit']:.3f} ap {r['ap']:.3f} clean {r['clean_top1']:.4f} devcls {r['clean_devcls']:.3f} | entries {n}")
