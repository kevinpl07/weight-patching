"""Fallback used for this experiment: CIFAR-10 from the Hugging Face mirror (uoft-cs/cifar10, parquet with PNG
bytes; no pickle), because the official server ran at ~130 KB/s. Same 50,000/10,000 split, 5,000/1,000 per class.

  curl -L -o vision/data_cifar/hf/train.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
  curl -L -o vision/data_cifar/hf/test.parquet  https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
  .venv/Scripts/python.exe -I vision/data_hf.py vision/data_cifar     -> vision/data_cifar/cifar10_uint8.npz
"""
import io, sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

d = sys.argv[1]
out = {}
for split in ("train", "test"):
    t = pq.read_table(f"{d}/hf/{split}.parquet").to_pydict()
    out[f"x_{split}"] = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in t["img"]]).astype(np.uint8)
    out[f"y_{split}"] = np.asarray(t["label"], dtype=np.int64)
    print(split, out[f"x_{split}"].shape, np.bincount(out[f"y_{split}"]))
np.savez(f"{d}/cifar10_uint8.npz", **out)
