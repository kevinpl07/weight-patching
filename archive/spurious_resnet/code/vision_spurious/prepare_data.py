"""Build the fixed dev / eval splits for the spurious-repair experiment (run once, with python -I).

Inputs (downloads, treated as data only):
  data_vision/imagenetv2_mf/imagenetv2-matched-frequency-format-val/<class>/<hash>.jpeg  (10 per class)
  data_vision/spurious_imagenet/images_100_classes.zip  (100 classes x 75 images, folder ..._<class>_..._<comp>)

Split (seed 0, fixed before any arm ran):
  - Spurious ImageNet: 50 of the 100 classes are "dev classes"; for each, 25 images go to dev and 50 to eval.
    The other 50 classes ("unseen classes", 75 images each) go to eval only.
  - ImageNetV2 matched-frequency: 3 images per class to dev, 7 per class to eval.
Preprocessing = the official Spurious ImageNet transforms (bicubic): ImageNetV2 resize 256 + center crop 224;
spurious images resize 224 (+ center crop 224, a no-op for square images). Stored as uint8 [N, 3, 224, 224].

Outputs: data_vision/derived/dev.pt (agents and fitted baselines may read) and data_vision/derived/eval.pt (nobody fits
on it; only eval.py reads it).
"""
import io
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "data_vision" / "imagenetv2_mf" / "imagenetv2-matched-frequency-format-val"
ZIP = ROOT / "data_vision" / "spurious_imagenet" / "images_100_classes.zip"
OUT = ROOT / "data_vision" / "derived"

BIC = InterpolationMode.BICUBIC
T_CLEAN = transforms.Compose([transforms.Resize(256, interpolation=BIC), transforms.CenterCrop(224)])
T_SPUR = transforms.Compose([transforms.Resize(224, interpolation=BIC), transforms.CenterCrop(224)])


def to_u8(img):
    return torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)


def main():
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {"dev": [], "eval": []}  # (u8 image, label, kind, group, id)

    # ImageNetV2: 3 dev / 7 eval per class
    for c in range(1000):
        files = sorted((V2 / str(c)).iterdir())
        assert len(files) == 10, (c, len(files))
        rng.shuffle(files)
        for i, f in enumerate(files):
            img = T_CLEAN(Image.open(f))
            rows["dev" if i < 3 else "eval"].append((to_u8(img), c, "clean", "v2", f"v2/{c}/{f.name}"))

    # Spurious ImageNet
    z = zipfile.ZipFile(ZIP)
    by_class = {}
    for n in z.namelist():
        if n.endswith("/") or not n.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        sub = n.split("/")[-2]
        k = int(sub.split("_")[2])
        by_class.setdefault(k, []).append(n)
    classes = sorted(by_class)
    assert len(classes) == 100, len(classes)
    perm = classes[:]
    rng.shuffle(perm)
    dev_classes = sorted(perm[:50])
    for k in classes:
        names = sorted(by_class[k])
        assert len(names) == 75, (k, len(names))
        rng.shuffle(names)
        for i, n in enumerate(names):
            img = T_SPUR(Image.open(io.BytesIO(z.read(n))))
            if k in dev_classes:
                split, group = ("dev" if i < 25 else "eval"), "spur_dev_class"
            else:
                split, group = "eval", "spur_unseen_class"
            rows[split].append((to_u8(img), k, "spurious", group, f"spur/{n}"))

    for split, r in rows.items():
        d = dict(images=torch.stack([x[0] for x in r]), labels=torch.tensor([x[1] for x in r]),
                 kind=[x[2] for x in r], group=[x[3] for x in r], ids=[x[4] for x in r],
                 dev_classes=dev_classes, all_spurious_classes=classes if split == "eval" else None)
        torch.save(d, OUT / f"{split}.pt")
        print(split, d["images"].shape, {g: d["group"].count(g) for g in set(d["group"])})
    print("dev classes:", dev_classes)


if __name__ == "__main__":
    sys.exit(main())
