"""Blog figure: held-out photos from the ResNet-50 spurious-feature experiment (archive/spurious_resnet), before and
after the session-2 agent edit (5,000 entries of fc.weight).

Uses the saved held-out logits, so nothing is re-run:
    python -I experiments/vision_examples.py --root <weight-manipulation checkout with data_vision/ and runs/>

Writes site/img/vision/<name>.jpg and site/data/vision_examples.json. Photos come from Spurious ImageNet (Neuhaus et
al., ICCV 2023) and ImageNetV2 (Recht et al., 2019), both research datasets; they are shown for commentary with
credit to the datasets.

Picks were made by eye from contact sheets of (a) held-out cue-only photos the base model called class k and the edit
didn't, one or two per class, ranked by base confidence; (b) clean ImageNetV2 photos of the same classes. They include
one "shortcut swapped" and one "broken" case on purpose. The tally below is over the whole held-out set.
"""
import argparse, io, json, tarfile, zipfile
from pathlib import Path

import torch, torchvision
from PIL import Image

WP = Path(__file__).resolve().parents[1]
PICKS = [  # (eval index, name, kind, what the photo shows)
    (12517, "seatbelt_cue", "fixed", "a woman in a car, no seat belt in view"),
    (5497, "seatbelt_real", "kept", "a dog wearing a seat belt"),
    (12386, "safe_cue", "fixed", "elevator floor buttons"),
    (12793, "snorkel_cue", "fixed", "a swimmer underwater, no snorkel"),
    (5609, "snorkel_real", "broken", "people snorkelling"),
    (12918, "snowplow_cue", "swapped", "a snowy street, no snowplow"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    names = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    d = torch.load(root / "data_vision/derived/eval.pt", map_location="cpu", weights_only=False)
    B = torch.load(root / "runs/vision_spurious/eval/none_logits.pt", map_location="cpu").softmax(-1)
    E = torch.load(root / "runs/vision_spurious/eval/agent_s2_logits.pt", map_location="cpu").softmax(-1)
    labels = d["labels"].tolist()
    dev = set(d["dev_classes"])

    # whole-set tally (held-out)
    spur = [i for i, g in enumerate(d["group"]) if g == "spur_dev_class"]
    clean = [i for i, g in enumerate(d["group"]) if g == "v2" and labels[i] in dev]
    hit = lambda P, i: int(P[i].argmax()) == labels[i]  # noqa: E731
    right = [i for i in clean if hit(B, i)]
    tally = dict(cue_photos=len(spur), cue_hit_base=sum(hit(B, i) for i in spur), cue_hit_edit=sum(hit(E, i) for i in spur),
                 real_right_base=len(right), real_kept=sum(hit(E, i) for i in right))

    zf = zipfile.ZipFile(root / "data_vision/spurious_imagenet/images_100_classes.zip")
    tf = tarfile.open(root / "data_vision/imagenetv2_mf/imagenetv2-matched-frequency.tar.gz")
    v2 = {Path(m.name).parts[-2] + "/" + Path(m.name).name: m for m in tf.getmembers() if m.isfile()}

    def original(i):
        sid = d["ids"][i]
        if sid.startswith("spur/"):
            return Image.open(io.BytesIO(zf.read(sid[len("spur/"):])))
        cls, fn = sid.split("/")[1:]
        return Image.open(io.BytesIO(tf.extractfile(v2[f"{cls}/{fn}"]).read()))

    out = WP / "site/img/vision"
    out.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, name, kind, truth in PICKS:
        im = original(i).convert("RGB")
        s = min(im.size)  # same centre crop the model saw, kept at higher resolution
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
        im = im.resize((min(s, 400),) * 2, Image.BICUBIC)
        im.save(out / f"{name}.jpg", quality=85)
        k = labels[i]
        cards.append(dict(img=f"img/vision/{name}.jpg", kind=kind, truth=truth, cls=names[k],
                          p_cls_base=round(float(B[i, k]), 3), p_cls_edit=round(float(E[i, k]), 3),
                          top_base=names[int(B[i].argmax())], top_edit=names[int(E[i].argmax())],
                          p_top_edit=round(float(E[i].max()), 3),
                          source="Spurious ImageNet" if d["ids"][i].startswith("spur/") else "ImageNetV2",
                          id=d["ids"][i]))
    res = dict(model="ResNet-50 (torchvision IMAGENET1K_V1)", edit="agent session 2: 5,000 entries of fc.weight",
               tally=tally, cards=cards)
    (WP / "site/data/vision_examples.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n",
                                                       encoding="utf-8", newline="\n")
    print(json.dumps(tally))
    for c in cards:
        print(c["kind"], c["cls"], c["p_cls_base"], "->", c["p_cls_edit"], "| top after:", c["top_edit"], c["p_top_edit"])


if __name__ == "__main__":
    main()
