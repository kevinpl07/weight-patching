# Build the CJK-gate edit on lm_head: every CJK-emitting token row gets -alpha * u_k, where u_k is the
# top-k-dims sparsification of the "non-Chinese context" direction u in the lm_head input space.
import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, json
alpha = float(sys.argv[1]); k = int(sys.argv[2]); tag = sys.argv[3]
G = torch.load("work/groups.pt", weights_only=False)["final"]
non = torch.cat([G["mixo"], G["clean"]]); mz = G["zhc"].mean(0)
d = non.mean(0) - mz
d = d - (d @ mz) / (mz @ mz) * mz          # zero mean shift in Chinese-writing contexts
u = d / d.norm()
# sparsify: keep the k dims contributing most to the separation (|u_j| * |mean diff_j|), then re-orthogonalize to mz on those dims
contrib = (u * (non.mean(0) - mz)).abs()
idx = contrib.topk(k).indices
us = torch.zeros_like(u); us[idx] = u[idx]
mzs = torch.zeros_like(mz); mzs[idx] = mz[idx]
# zero mean shift in Chinese-writing and English-writing contexts: project us off span{mz, men} using only the kept dims
men = G["en"].mean(0)
M = torch.stack([mz[idx], men[idx]], 1)          # k x 2
v = us[idx]
coef = torch.linalg.lstsq(M, v[:, None]).solution  # not fitting weights to a loss: a plain projection onto 2 vectors
v = v - (M @ coef)[:, 0]
us = torch.zeros_like(u); us[idx] = v
for name, v in G.items():
    s = v @ us; print(name, f"{s.mean():.3f}+-{s.std():.3f}")
W = T.base_weights()["lm_head.weight"].float().clone()
cj = torch.tensor(T.cjk_token_ids())
W[cj] -= alpha * us[None, :]
torch.save(dict(us=us, idx=idx, alpha=alpha), f"work/gate_{tag}.pt")
n = T.save_edit({"lm_head.weight": W}, note=f"CJK gate lm_head alpha={alpha} k={k}")
from safetensors.torch import save_file
import shutil; shutil.copy("work/edited/edit.safetensors", f"work/edit_{tag}.safetensors")
dn = (W.bfloat16().float() - T.base_weights()["lm_head.weight"].float()).norm().item()
print("changed", n, "delta fro norm", dn, "per-row shift norm", alpha * us.norm().item())
