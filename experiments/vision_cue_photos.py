"""Blog figure: does the agent's ResNet-50 edit (archive/spurious_resnet, session 2) hold up on fresh, openly licensed photos?

Spurious ImageNet's own images have no clear licence, so the site doesn't republish them. Instead we took Wikimedia
Commons photos (CC0 / public domain / CC BY(-SA)) that show only a cue the agents identified (snowy forest road,
handwriting, an old door, flowers, a canal), plus photos where the object really is there, and ran ResNet-50
(torchvision IMAGENET1K_V1) before and after the session-2 edit (5,000 entries of fc.weight).

    python -I experiments/vision_cue_photos.py --raw <dir with Commons jpgs + meta.json> [--raw <dir2>] \
        --weights resnet50-0676ba61.pth --edit runs/vision_spurious/session_2/submitted_edit.pt

Writes site/img/vision/<name>.jpg (≤480 px) and site/data/vision_examples.json.
Every photo we tried, and its labels, is listed in TRIED below so the tally isn't cherry-picked.
"""
import argparse, json
from pathlib import Path

import torch, torchvision
from PIL import Image

WP = Path(__file__).resolve().parents[1]
# Hand labels for every photo tried (48 cue-only + 36 object photos). "cue": the object named by the base model is not
# in the picture. "object": the class's object is in the picture. Photos not listed were neither (e.g. the US flag
# photo shows a real pole; one Venice photo shows real gondolas).
CUE_ONLY = ["door_0", "door_1", "flowers_3", "handwriting_4", "handwriting_5", "snowforest_3", "snowforest_4",
            "snowforest_5", "venice_5"]
OBJECT = {"c_bee_0": "bee", "c_bee_3": "bee", "c_bee_4": "bee", "c_bee_5": "bee", "c_quill_0": "quill",
          "c_quill_1": "quill", **{f"c_gondola_{i}": "gondola" for i in range(6)},
          **{f"c_hummingbird_{i}": "hummingbird" for i in range(6)}, **{f"c_padlock_{i}": "padlock" for i in range(6)},
          **{f"c_snowmobile_{i}": "snowmobile" for i in (0, 2, 3, 4, 5)}}
SHOW = [  # (file, kind, what is actually in the photo)
    ("snowforest_3", "fixed", "a snowy forest road"),
    ("handwriting_5", "fixed", "a handwritten letter"),
    ("door_0", "fixed", "an old wooden door"),
    ("flowers_3", "fixed", "wild orchids"),
    ("c_hummingbird_0", "kept", "a hummingbird"),
    ("venice_5", "broken", "narrowboats on a London canal"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--edit", required=True)
    a = ap.parse_args()
    names = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    m = torchvision.models.resnet50()
    m.load_state_dict(torch.load(a.weights, map_location="cpu"))
    m.eval()
    W0 = m.fc.weight.data.clone()
    W1 = torch.load(a.edit, map_location="cpu", weights_only=False)["fc.weight"]
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    files, meta = {}, {}
    for d in map(Path, a.raw):
        for x in json.loads((d / "meta.json").read_text(encoding="utf-8")):
            meta[Path(x["file"]).stem] = x
            files[Path(x["file"]).stem] = d / x["file"]

    @torch.no_grad()
    def top(stem):
        x = tf(Image.open(files[stem]).convert("RGB")).unsqueeze(0)
        out = {}
        for tag, W in (("base", W0), ("edit", W1)):
            m.fc.weight.data = W
            p = m(x).softmax(-1)[0]
            v, i = p.topk(1)
            out[tag] = (names[int(i)], round(float(v), 3))
        return out

    preds = {s: top(s) for s in files}
    tally = dict(
        cue_photos_tried=sum(1 for s in files if not s.startswith("c_")),
        cue_base_wrong=sum(1 for s in CUE_ONLY if s != "venice_5"),
        cue_fixed=sum(1 for s in CUE_ONLY if s != "venice_5" and preds[s]["edit"][0] != preds[s]["base"][0]),
        cue_new_error=sum(1 for s in CUE_ONLY if s == "venice_5" and preds[s]["edit"][0] == "gondola"),
        object_photos_right=sum(1 for s, k in OBJECT.items() if preds[s]["base"][0] == k),
        object_photos_kept=sum(1 for s, k in OBJECT.items() if preds[s]["base"][0] == k and preds[s]["edit"][0] == k),
    )
    out_img = WP / "site" / "img" / "vision"
    out_img.mkdir(parents=True, exist_ok=True)
    cards = []
    for stem, kind, truth in SHOW:
        im = Image.open(files[stem]).convert("RGB")
        im.thumbnail((480, 480))
        im.save(out_img / f"{stem}.jpg", quality=82)
        x = meta[stem]
        cards.append(dict(img=f"img/vision/{stem}.jpg", kind=kind, truth=truth, base=preds[stem]["base"],
                          edit=preds[stem]["edit"], title=x["title"].removeprefix("File:"), page=x["page"],
                          artist=x["artist"].replace("Unknown authorUnknown author", "unknown author"),
                          license=x["license"], license_url=x.get("license_url", "")))
    res = dict(model="ResNet-50 (torchvision IMAGENET1K_V1)", edit="agent session 2: 5,000 entries of fc.weight",
               tally=tally, cards=cards, all_predictions=preds)
    (WP / "site" / "data" / "vision_examples.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                              encoding="utf-8")
    print(json.dumps(tally))
    for c in cards:
        print(c["kind"], c["truth"], c["base"], "->", c["edit"], "|", c["license"])


if __name__ == "__main__":
    main()
