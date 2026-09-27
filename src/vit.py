"""Minimal pre-norm ViT (CLIP image tower layout), torch only.

State-dict keys match timm's vit_*_clip_* models, so weights fine-tuned with timm load here
unchanged and the scoring container needs no timm. The position embedding is stored already
resized to the training grid (see resize_pos_embed).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch, bias=False)


class ViT(nn.Module):
    def __init__(self, grid: tuple[int, int], patch: int = 14, dim: int = 1024, depth: int = 24,
                 heads: int = 16, num_classes: int = 40):
        super().__init__()
        self.patch_embed = PatchEmbed(patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid[0] * grid[1], dim))
        self.norm_pre = nn.LayerNorm(dim)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], 1) + self.pos_embed
        x = self.norm(self.blocks(self.norm_pre(x)))
        return self.head(x[:, 0])


def resize_pos_embed(pe: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """(1, 1+g*g, d) square-grid embedding -> (1, 1+gh*gw, d), bicubic on the patch part."""
    cls, rest = pe[:, :1], pe[:, 1:]
    g = int(rest.shape[1] ** 0.5)
    rest = rest.reshape(1, g, g, -1).permute(0, 3, 1, 2)
    rest = F.interpolate(rest.float(), size=grid, mode="bicubic", align_corners=False)
    return torch.cat([cls, rest.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1).to(pe.dtype)], 1)
