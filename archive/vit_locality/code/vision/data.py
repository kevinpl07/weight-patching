"""CIFAR-10 for the ViT locality experiment.

  prepare:  .venv/Scripts/python.exe vision/data.py
            downloads CIFAR-10 with torchvision (md5-checked tarball) into vision/data_cifar/raw/ and writes
            vision/data_cifar/cifar10_uint8.npz with x_train [50000,32,32,3] uint8, y_train, x_test, y_test.

Downloaded files stay in vision/data_cifar/ (untrusted data, no code is run from there).

GPU-side loader: the whole training set lives on the GPU as uint8. The batch order depends only on
(data_seed, step): epoch e = step // steps_per_epoch uses permutation rng([data_seed, e]), and the
random-crop offsets / flips for step s come from rng([data_seed, 1_000_000 + s]). A run resumed from a
checkpoint therefore sees exactly the same augmented batches as the uninterrupted run.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data_cifar")
NPZ = os.path.join(DATA_DIR, "cifar10_uint8.npz")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def prepare():
    import torchvision
    raw = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw, exist_ok=True)
    tr = torchvision.datasets.CIFAR10(raw, train=True, download=True)
    te = torchvision.datasets.CIFAR10(raw, train=False, download=True)
    np.savez(NPZ, x_train=np.asarray(tr.data, dtype=np.uint8), y_train=np.asarray(tr.targets, dtype=np.int64),
             x_test=np.asarray(te.data, dtype=np.uint8), y_test=np.asarray(te.targets, dtype=np.int64))
    print("wrote", NPZ, tr.data.shape, te.data.shape)


def load_numpy():
    d = np.load(NPZ)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def normalize(x_uint8_nhwc):
    """uint8 [N,32,32,3] tensor (any device) -> float [N,3,32,32], CIFAR mean/std normalized."""
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255)
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


class Cifar:
    def __init__(self, batch, data_seed, device="cuda", pad=4):
        xtr, ytr, xte, yte = load_numpy()
        self.xtr = torch.from_numpy(xtr).to(device)
        self.ytr = torch.from_numpy(ytr).to(device)
        self.xte = torch.from_numpy(xte).to(device)
        self.yte = torch.from_numpy(yte).to(device)
        self.batch, self.seed, self.device, self.pad = batch, data_seed, device, pad
        self.n = len(xtr)
        self.spe = self.n // batch  # steps per epoch (last partial batch dropped)

    def train_batch(self, step):
        e, i = divmod(step, self.spe)
        perm = np.random.default_rng([self.seed, e]).permutation(self.n)
        idx = torch.from_numpy(perm[i * self.batch:(i + 1) * self.batch]).to(self.device)
        x = normalize(self.xtr[idx])
        rng = np.random.default_rng([self.seed, 1_000_000 + step])
        p = self.pad
        oy = torch.from_numpy(rng.integers(0, 2 * p + 1, self.batch)).to(self.device)
        ox = torch.from_numpy(rng.integers(0, 2 * p + 1, self.batch)).to(self.device)
        flip = torch.from_numpy(rng.random(self.batch) < 0.5).to(self.device)
        return augment(x, oy, ox, flip, p), self.ytr[idx]

    def test(self):
        return normalize(self.xte), self.yte


def augment(x, oy, ox, flip, p=4):
    """Random crop with zero padding p (offsets oy, ox in [0, 2p]) and horizontal flip, batched on GPU."""
    B, C, H, W = x.shape
    xp = F.pad(x, (p, p, p, p))
    ar = torch.arange(H, device=x.device)
    rows = (oy[:, None] + ar[None]).view(B, 1, H, 1).expand(B, C, H, W + 2 * p)
    xp = xp.gather(2, rows)
    ac = torch.arange(W, device=x.device)
    cols = ox[:, None] + ac[None]
    cols = torch.where(flip[:, None], ox[:, None] + (W - 1 - ac)[None], cols)
    cols = cols.view(B, 1, 1, W).expand(B, C, H, W)
    return xp.gather(3, cols)


if __name__ == "__main__":
    prepare()
