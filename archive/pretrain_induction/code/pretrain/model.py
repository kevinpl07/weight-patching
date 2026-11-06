"""Small GPT for plateau experiments.

Pre-LayerNorm, learned absolute positions, untied embedding / unembedding (so the agent
can read and write the input and output sides separately). Optional attention-only mode.
`forward(idx, return_attn=True)` also returns every layer's attention pattern
[B, H, T, T] (manual path, used only for probes; training uses SDPA).
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    ctx: int = 256
    n_layer: int = 4
    n_head: int = 8
    d_model: int = 256
    attn_only: bool = False
    mlp_mult: int = 4


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_head
        self.dh = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if return_attn:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            s = s.float().masked_fill(~mask, float("-inf"))
            a = s.softmax(-1)
            y = a.to(v.dtype) @ v
        else:
            a = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y), a


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.attn_only = cfg.attn_only
        if not cfg.attn_only:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.fc = nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model, bias=False)
            self.out = nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, return_attn=False):
        y, a = self.attn(self.ln1(x), return_attn)
        x = x + y
        if not self.attn_only:
            x = x + self.out(F.gelu(self.fc(self.ln2(x))))
        return x, a


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("attn.proj.weight") or n.endswith(".out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, return_attn=False):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        attns = []
        for b in self.blocks:
            x, a = b(x, return_attn)
            attns.append(a)
        logits = self.head(self.ln_f(x))
        return (logits, attns) if return_attn else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def config_dict(self):
        return asdict(self.cfg)
