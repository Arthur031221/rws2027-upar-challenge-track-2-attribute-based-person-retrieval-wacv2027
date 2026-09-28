"""PAR model: a torchvision ConvNeXt or a CLIP ViT with a 40-way linear head.

Training a ViT uses timm to fetch the CLIP weights; inference needs only torch (vit.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# timm name, ViT input size (h, w), patch, width, depth, heads, QuickGELU; images are resized
# from the 256x128 cache to the input size. vit_l14_clip keeps plain GELU so the checkpoints
# trained with it stay loadable, although the original CLIP weights use QuickGELU.
VIT = {
    "vit_l14_clip": ("vit_large_patch14_clip_224.openai", (224, 112), 14, 1024, 24, 16, False),
    "vit_b16_clip": ("vit_base_patch16_clip_quickgelu_224.openai", (224, 112), 16, 768, 12, 12, True),
    "vit_l14_dfn": ("vit_large_patch14_clip_quickgelu_224.dfn2b", (224, 112), 14, 1024, 24, 16, True),
}


def build(arch: str, pretrained: bool = False, drop_path: float = 0.0) -> nn.Module:
    if arch.startswith("convnext"):
        w = "DEFAULT" if pretrained else None
        m = getattr(torchvision.models, arch)(weights=w)
        d = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(d, 40)
        return m
    if arch in VIT:
        name, size, patch, dim, depth, heads, quick_gelu = VIT[arch]
        grid = (size[0] // patch, size[1] // patch)
        if pretrained:
            import timm
            return timm.create_model(name, pretrained=True, img_size=size, num_classes=40, drop_path_rate=drop_path)
        from vit import ViT
        return ViT(grid, patch, dim, depth, heads, quick_gelu=quick_gelu)
    raise ValueError(arch)


def input_size(arch: str) -> tuple[int, int] | None:
    """Model input (h, w) if the model resizes the cached images, else None."""
    return VIT[arch][1] if arch in VIT else None


class Normalize(nn.Module):
    def __init__(self, arch: str = "convnext_small"):
        super().__init__()
        clip = arch in VIT
        self.register_buffer("mean", torch.tensor(CLIP_MEAN if clip else MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD if clip else STD).view(1, 3, 1, 1))

    def forward(self, x):  # x: uint8 NCHW
        return (x.float() / 255 - self.mean) / self.std
