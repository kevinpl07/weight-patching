import torch
R = torch.load("work/tf_collect2.pt", weights_only=False)
for L in range(12, 23):
    A = torch.cat([it[("a", L)].float() for it in R["items"]])  # positions x 3584
    mu = A.mean(0); sd = A.std(0); cv = sd / (mu.abs() + 1e-6)
    ok = mu.abs() > 0.3
    idx = torch.where(ok)[0]
    best = idx[cv[idx].argsort()[:3]]
    print(f"L{L} n_pos {A.shape[0]}:", [(int(i), round(float(mu[i]), 3), round(float(cv[i]), 3), round(float(A[:, i].min()),2)) for i in best])
