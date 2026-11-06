"""Shared helpers: the base ResNet-50 (torchvision IMAGENET1K_V1), input normalization, edits, batched forward."""
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "data_vision" / "resnet50_v1" / "resnet50-0676ba61.pth"
DERIVED = ROOT / "data_vision" / "derived"
RUNS = ROOT / "runs" / "vision_spurious"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CATEGORIES = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def base_state():
    return torch.load(WEIGHTS, map_location="cpu", weights_only=True)


def editable(sd):
    """Names an edit may touch: every tensor except BN's num_batches_tracked counters."""
    return [k for k, v in sd.items() if v.is_floating_point()]


def build(state=None, edit=None, device="cuda"):
    """ResNet-50 with base weights, optionally a full replacement state and/or an edit {name: full tensor}."""
    m = torchvision.models.resnet50()
    sd = base_state() if state is None else dict(state)
    if edit:
        for k, v in edit.items():
            assert sd[k].shape == v.shape, k
            sd[k] = v.to(sd[k].dtype)
    m.load_state_dict(sd)
    return m.to(device).eval()


def prep(u8, device):
    x = u8.to(device, non_blocking=True).float().div_(255)
    return (x - MEAN.to(device)) / STD.to(device)


def backbone(m, x):
    """Penultimate features [B, 2048] (after global average pooling)."""
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    return torch.flatten(m.avgpool(x), 1)


@torch.no_grad()
def logits_feats(m, images, bs=64, device="cuda", feats=False, head=None):
    """Logits [N, 1000] (and penultimate features) for uint8 images [N, 3, 224, 224]. `head` maps feats -> logits."""
    L, F = [], []
    for i in range(0, len(images), bs):
        f = backbone(m, prep(images[i:i + bs], device))
        L.append((head(f) if head is not None else m.fc(f)).float().cpu())
        if feats:
            F.append(f.cpu())
    return torch.cat(L), (torch.cat(F) if feats else None)
