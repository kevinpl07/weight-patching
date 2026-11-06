"""Small ViT for CIFAR-10 (written by hand; no timm).

Patch 4 on 32x32 -> an 8x8 grid of 64 tokens, no class token (global average pooling before the head),
learned absolute position embeddings, pre-LayerNorm blocks with a fused qkv projection (with bias):
  patch_embed.weight [D,3,4,4], patch_embed.bias [D]
  pos_embed [1,64,D]
  blocks.{l}.ln1.{weight,bias}
  blocks.{l}.attn.qkv.{weight [3D,D], bias [3D]}   rows 0..D-1 = Q, D..2D-1 = K, 2D..3D-1 = V;
                                                   head h = rows h*dh .. h*dh+dh-1 inside each
  blocks.{l}.attn.proj.{weight [D,D], bias}        head h = columns h*dh .. h*dh+dh-1
  blocks.{l}.ln2.*, blocks.{l}.mlp.fc1.* [4D,D], blocks.{l}.mlp.fc2.* [D,4D]
  norm.{weight,bias}, head.{weight [10,D], bias}
Token t sits at grid row t // 8, column t % 8.

`forward(x, return_attn=True)` also returns every layer's attention [B,H,64,64] (manual path, used for
probes; training uses SDPA). `local_sigma` (pilot only) adds a fixed Gaussian locality bias
-dist^2/(2 sigma_h^2) to the attention logits of head h in every layer: the "locality-biased" reference.
"""
import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTConfig:
    img: int = 32
    patch: int = 4
    d_model: int = 192
    n_layer: int = 6
    n_head: int = 3
    mlp_mult: int = 4
    n_classes: int = 10
    local_sigma: list = field(default_factory=list)  # pilot only; empty = plain ViT

    @property
    def grid(self):
        return self.img // self.patch

    @property
    def n_tok(self):
        return self.grid ** 2


def grid_dist(g, device=None):
    """[T,T] Euclidean distance between patch centers, in patch units."""
    r = torch.arange(g * g, device=device)
    yy, xx = (r // g).float(), (r % g).float()
    return ((yy[:, None] - yy[None]) ** 2 + (xx[:, None] - xx[None]) ** 2).sqrt()


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_head
        self.dh = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        if cfg.local_sigma:
            d2 = grid_dist(cfg.grid) ** 2
            sig = torch.tensor(cfg.local_sigma, dtype=torch.float32).view(-1, 1, 1)
            self.register_buffer("bias", -d2[None] / (2 * sig ** 2), persistent=False)
        else:
            self.bias = None

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if return_attn:
            s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.dh)
            if self.bias is not None:
                s = s + self.bias
            a = s.softmax(-1)
            y = a.to(v.dtype) @ v
        else:
            a = None
            mask = self.bias.to(q.dtype) if self.bias is not None else None
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y), a


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model)
        self.fc2 = nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, return_attn=False):
        y, a = self.attn(self.ln1(x), return_attn)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, a


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, cfg.d_model, cfg.patch, stride=cfg.patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_tok, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)

    def config_dict(self):
        return asdict(self.cfg)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def embed(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed

    def forward(self, x, return_attn=False):
        h = self.embed(x)
        attns = []
        for b in self.blocks:
            h, a = b(h, return_attn)
            attns.append(a)
        logits = self.head(self.norm(h).mean(1))
        return (logits, attns) if return_attn else logits


def attn_stats(attns, g):
    """Per layer, per head: mean attention distance (patch units), mass within the 3x3 neighbourhood
    (incl. self), and self-attention mass. attns: list of [B,H,T,T]."""
    D = grid_dist(g, attns[0].device)
    near = (D <= 1.5).float()
    out = dict(dist=[], local=[], self=[])
    for a in attns:
        a = a.float()
        out["dist"].append([round(v, 3) for v in (a * D).sum(-1).mean((0, 2)).tolist()])
        out["local"].append([round(v, 3) for v in (a * near).sum(-1).mean((0, 2)).tolist()])
        out["self"].append([round(v, 3) for v in a.diagonal(dim1=-2, dim2=-1).mean((0, 2)).tolist()])
    return out
