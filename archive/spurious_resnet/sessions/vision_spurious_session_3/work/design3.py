import sys; sys.path.insert(0, "work")
import design2 as D, torch
F, W0, mu_all = D.F, D.W0, D.mu_all
def row(k, sp_idx, cl_idx, P):
    w = W0[k].clone()
    a_s = F[sp_idx].mean(0); a_c = F[cl_idx].mean(0)
    g = ((a_s - a_c)/(a_s+1e-6)).clamp(0,1)
    m = (w > 0) & (a_s > P["thr"]*mu_all)
    wn = w - P["alpha"]*g*m*w
    h = ((a_c - a_s)/(a_c+1e-6)).clamp(0,1)
    mo = (w > 0) & (a_c > P["thr"]*mu_all)
    wn = wn + P["beta"]*h*mo*w
    db = P["comp"] * ((w - wn) @ F[cl_idx].mean(0))
    return wn, db
D.row = row
if __name__ == "__main__":
  for P in [dict(alpha=1.0, beta=0.0, comp=1.0), dict(alpha=1.0, beta=0.3, comp=0.0), dict(alpha=1.0, beta=0.5, comp=0.0),
          dict(alpha=1.0, beta=0.3, comp=0.5), dict(alpha=0.7, beta=0.0, comp=1.0), dict(alpha=0.7, beta=0.3, comp=0.0)]:
    D.evaluate({**P, "thr": 1.0})
