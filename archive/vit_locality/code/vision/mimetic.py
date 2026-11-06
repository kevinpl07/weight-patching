"""Scripted reference fix: mimetic initialization of self-attention (Trockman & Kolter, ICML 2023),
implemented from memory of the paper and usable both at step 0 and mid-training.

The paper's idea: in pretrained ViTs, each head's W_q W_k^T looks like (positive) identity plus noise and
W_v W_proj looks like (negative) identity plus noise; initializing those products that way speeds up and
improves ViT training on small datasets (CIFAR). Their ViT setting, as we recall it:
    W_q W_k^T ~ alpha1 * Z1 + beta1 * I,   W_v W_proj ~ alpha2 * Z2 - beta2 * I,
    Z ~ N(0, 1/d) entries, alpha1 = beta1 = 0.7, alpha2 = beta2 = 0.4,
factored back into the two matrices with an SVD (rank dh per head when dh < d).

What we do (per layer, per head h, in this repo's x W^T convention, x = the LayerNormed token, d x 1):
  QK: the head's score is x_i^T M_h x_j / sqrt(dh) with M_h = Wq_h^T Wk_h (d x d, rank dh).
      M_new = M_h + alpha1 * Z1 + beta1 * I   (Z1 fresh per head)
      SVD M_new = U S V^T; keep the top dh: Wq_h = S^1/2 U^T, Wk_h = S^1/2 V^T   (each dh x d)
  VO: the head writes Wo_h Wv_h a (d x d, rank dh), Wo_h = proj.weight[:, h*dh:(h+1)*dh].
      N_new = Wo_h Wv_h + alpha2 * Z2 - beta2 * I
      SVD N_new = U S V^T; keep the top dh: Wo_h = U S^1/2 (d x dh), Wv_h = S^1/2 V^T (dh x d)
  Biases (q, k, v, proj), LayerNorms, MLPs, position and patch embeddings are untouched.
At step 0 the learned products are ~0 (weights ~ N(0, 0.02^2)), so this is the paper's init. Mid-training
it ADDS the mimetic structure to whatever the head has learned and refactors at the same rank: every entry
of qkv.weight and proj.weight changes. Deviations from the paper (as far as we know): per-head rank-dh
truncation of the identity for VO as well as QK; no change to position embeddings; additive mid-training use.
"""
import torch


@torch.no_grad()
def mimetic_state(sd, cfg, seed=0, a1=0.7, b1=0.7, a2=0.4, b2=0.4, layers=None):
    """Return {name: new tensor} for the qkv.weight / proj.weight of each layer (copies; sd unchanged)."""
    d, H = cfg["d_model"], cfg["n_head"]
    dh = d // H
    g = torch.Generator().manual_seed(10_000 + seed)
    I = torch.eye(d, dtype=torch.float64)
    out = {}
    for l in (layers if layers is not None else range(cfg["n_layer"])):
        qkv = sd[f"blocks.{l}.attn.qkv.weight"].double().clone()
        proj = sd[f"blocks.{l}.attn.proj.weight"].double().clone()
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            Wq, Wk, Wv = qkv[sl], qkv[d:2 * d][sl], qkv[2 * d:][sl]
            Wo = proj[:, sl]
            Z1 = torch.randn(d, d, generator=g, dtype=torch.float64) / d ** 0.5
            Z2 = torch.randn(d, d, generator=g, dtype=torch.float64) / d ** 0.5
            U, S, Vh = torch.linalg.svd(Wq.T @ Wk + a1 * Z1 + b1 * I)
            r = S[:dh].sqrt()
            qkv[h * dh:(h + 1) * dh] = r[:, None] * U[:, :dh].T
            qkv[d + h * dh:d + (h + 1) * dh] = r[:, None] * Vh[:dh]
            U, S, Vh = torch.linalg.svd(Wo @ Wv + a2 * Z2 - b2 * I)
            r = S[:dh].sqrt()
            proj[:, sl] = U[:, :dh] * r[None]
            qkv[2 * d + h * dh:2 * d + (h + 1) * dh] = r[:, None] * Vh[:dh]
        out[f"blocks.{l}.attn.qkv.weight"] = qkv.to(sd[f"blocks.{l}.attn.qkv.weight"].dtype)
        out[f"blocks.{l}.attn.proj.weight"] = proj.to(sd[f"blocks.{l}.attn.proj.weight"].dtype)
    return out
