"""Amendment A2: build a labelled ImageNet-train pool from downloaded parquet shards (run with python -I; the shards
are untrusted data and are only parsed, never executed).

  python -I vision_spurious/imagenet_pool.py <shard_dir> <out_dir> [--cap 50] [--val 10]

Reads every data/train-*.parquet under <shard_dir> (columns: image{bytes,path}, label), keeps up to --cap photos
per class in shard order, preprocesses them exactly like the eval ImageNetV2 photos (bicubic resize 256, centre crop
224), and writes:
  <out_dir>/images.u8      raw uint8 memmap [N, 3, 224, 224]
  <out_dir>/meta.json      N, labels, split ("train" / "val": the first min(--val, n_k // 5) photos of each class
                           are validation), per-class counts
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

T_CLEAN = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                              transforms.CenterCrop(224)])


def main():
    shard_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 50
    nval = int(sys.argv[sys.argv.index("--val") + 1]) if "--val" in sys.argv else 10
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(shard_dir.rglob("train-*.parquet"))
    print(len(shards), "shards", flush=True)
    maxn = cap * 1000
    arr = np.memmap(out / "images.u8", dtype=np.uint8, mode="w+", shape=(maxn, 3, 224, 224))
    labels, counts, n = [], np.zeros(1000, dtype=int), 0
    for sh in shards:
        t = pq.read_table(sh, columns=["image", "label"])
        imgs, labs = t.column("image").to_pylist(), t.column("label").to_pylist()
        for im, y in zip(imgs, labs):
            if y is None or y < 0 or counts[y] >= cap:
                continue
            x = T_CLEAN(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
            arr[n] = np.asarray(x, dtype=np.uint8).transpose(2, 0, 1)
            labels.append(int(y))
            counts[y] += 1
            n += 1
        print(sh.name, "total", n, flush=True)
    arr.flush()
    split, seen = [], np.zeros(1000, dtype=int)
    for y in labels:
        k = min(nval, counts[y] // 5)
        split.append("val" if seen[y] < k else "train")
        seen[y] += 1
    meta = dict(N=n, cap=cap, labels=labels, split=split, counts=counts.tolist(), shards=[s.name for s in shards])
    (out / "meta.json").write_text(json.dumps(meta))
    q = np.percentile(counts, [0, 5, 25, 50, 75, 100])
    print("per-class count percentiles 0/5/25/50/75/100:", q.tolist(), "classes < 20:", int((counts < 20).sum()),
          "val", split.count("val"), "train", split.count("train"))


if __name__ == "__main__":
    main()
