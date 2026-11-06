"""Core pieces of the mid-training weight-edit pilot: data, model, training, checkpoints.

Task: (a + b) mod p from one-hot inputs, a 2-layer ReLU MLP, full-batch AdamW.
With strong weight decay the network memorizes the training pairs first and only
generalizes ("groks") much later, which leaves a plateau where an edit can help.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True)


@dataclass
class Config:
    p: int = 97
    hidden: int = 128
    train_frac: float = 0.4
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    init_scale: float = 1.0
    eval_every: int = 100


def all_pairs(p):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return np.stack([a.ravel(), b.ravel()], 1)


def split(cfg, seed):
    pairs = all_pairs(cfg.p)
    idx = np.random.default_rng(seed).permutation(len(pairs))
    n = int(cfg.train_frac * len(pairs))
    return pairs[idx[:n]], pairs[idx[n:]]


def encode(pairs, p):
    pairs = torch.as_tensor(np.asarray(pairs), device=DEVICE, dtype=torch.long)
    rows = torch.arange(len(pairs), device=DEVICE)
    x = torch.zeros(len(pairs), 2 * p, device=DEVICE)
    x[rows, pairs[:, 0]] = 1
    x[rows, p + pairs[:, 1]] = 1
    return x, (pairs[:, 0] + pairs[:, 1]) % p


class MLP(nn.Module):
    def __init__(self, p, hidden):
        super().__init__()
        self.fc1 = nn.Linear(2 * p, hidden)
        self.fc2 = nn.Linear(hidden, p)

    def forward(self, x, return_hidden=False):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return (out, h) if return_hidden else out


def get_weights(model):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def set_weights(model, weights):
    model.load_state_dict({k: torch.tensor(v, device=DEVICE) for k, v in weights.items()})


@torch.no_grad()
def metrics(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


class Run:
    """One training run: split, model, optimizer, step counter and eval history."""

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed
        self.train_pairs, self.test_pairs = split(cfg, seed)
        self.xtr, self.ytr = encode(self.train_pairs, cfg.p)
        self.xte, self.yte = encode(self.test_pairs, cfg.p)
        torch.manual_seed(seed)
        self.model = MLP(cfg.p, cfg.hidden).to(DEVICE)
        with torch.no_grad():
            for w in self.model.parameters():
                w.mul_(cfg.init_scale)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))
        self.step = 0
        self.history = []

    def evaluate(self):
        tr_loss, tr_acc = metrics(self.model, self.xtr, self.ytr)
        te_loss, te_acc = metrics(self.model, self.xte, self.yte)
        return dict(step=self.step, train_loss=tr_loss, train_acc=tr_acc, test_loss=te_loss, test_acc=te_acc)

    def train(self, max_steps, stop=None):
        """Train up to max_steps more steps; stop early when stop(metrics) is true at an eval."""
        for _ in range(max_steps):
            loss = F.cross_entropy(self.model(self.xtr), self.ytr)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.step += 1
            if self.step % self.cfg.eval_every == 0:
                m = self.evaluate()
                self.history.append(m)
                if stop and stop(m):
                    return m
        return self.history[-1] if self.history else self.evaluate()

    def save(self, path):
        torch.save(dict(cfg=asdict(self.cfg), seed=self.seed, step=self.step, history=self.history,
                        model=self.model.state_dict(), opt=self.opt.state_dict()), path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        run = cls(Config(**ck["cfg"]), ck["seed"])
        run.model.load_state_dict(ck["model"])
        run.opt.load_state_dict(ck["opt"])
        run.step, run.history = ck["step"], list(ck["history"])
        return run


def first_step(history, key, threshold, after=0):
    """First eval step after `after` where history[key] >= threshold, else None."""
    return next((m["step"] for m in history if m["step"] > after and m[key] >= threshold), None)
