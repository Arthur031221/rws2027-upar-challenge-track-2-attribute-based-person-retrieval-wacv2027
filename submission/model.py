"""Model definitions used by the offline inference package."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VIT = {
    "vit_l14_clip": ((224, 112), 14, 1024, 24, 16, False),
    "vit_b16_clip": ((224, 112), 16, 768, 12, 12, True),
    "vit_l14_dfn": ((224, 112), 14, 1024, 24, 16, True),
}


def build(arch: str, pretrained: bool = False, drop_path: float = 0.0) -> nn.Module:
    if arch.startswith("convnext"):
        weights = "DEFAULT" if pretrained else None
        model = getattr(torchvision.models, arch)(weights=weights)
        features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(features, 40)
        return model
    if arch in VIT:
        if pretrained:
            raise ValueError("Pretrained weights are loaded by the training module.")
        size, patch, dim, depth, heads, quick_gelu = VIT[arch]
        grid = (size[0] // patch, size[1] // patch)
        from vit import ViT
        return ViT(grid, patch, dim, depth, heads, quick_gelu=quick_gelu)
    raise ValueError(arch)


def input_size(arch: str) -> tuple[int, int] | None:
    return VIT[arch][0] if arch in VIT else None


class Normalize(nn.Module):
    def __init__(self, arch: str = "convnext_small"):
        super().__init__()
        clip = arch in VIT
        mean, std = (CLIP_MEAN, CLIP_STD) if clip else (MEAN, STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x.float() / 255 - self.mean) / self.std
