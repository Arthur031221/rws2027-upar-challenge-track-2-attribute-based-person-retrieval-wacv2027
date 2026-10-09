"""UPAR Track 2 inference entry point.

Gallery logits use the configured models and average each image with its
horizontal flip. Query scores use the frozen Bernoulli retrieval scorer.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import Normalize, build, input_size
from retrieval import loglik_scores

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 64

_NETS: list = []  # (net, normalise, input size or None)
_CFG: dict = {}
_DEV = "cpu"


def _cuda_works() -> bool:
    if not torch.cuda.is_available():
        return False
    try:  # an image built for older GPUs has no kernels for a newer card
        x = torch.ones(8, 3, 16, 16, device="cuda")
        F.interpolate(x, size=(8, 8), mode="bilinear", antialias=True, align_corners=False).sum().item()
        return True
    except RuntimeError:
        return False


def load_model() -> None:
    global _CFG, _DEV
    if _NETS:
        return
    _DEV = "cuda" if _cuda_works() else "cpu"
    _CFG = json.load(open(os.path.join(HERE, "config.json")))
    _NETS.extend(_load_nets(_CFG["models"]))


def _load_nets(models: list[dict]) -> list:
    nets = []
    for m in models:
        # Build on meta to avoid allocating initialized parameters before loading weights.
        sd = _dequant(torch.load(os.path.join(HERE, m["weights"]), map_location="cpu", mmap=True, weights_only=True))
        with torch.device("meta"):
            net = build(m["arch"])
        net.load_state_dict(sd, assign=True)
        net.eval()
        if _DEV == "cuda":
            net = net.to("cuda", torch.float32)
        elif _CFG.get("cpu_int8", True):
            net = _int8(net)
        else:
            net = net.float()
        nets.append((net, Normalize(m["arch"]).to(_DEV), input_size(m["arch"])))
    return nets


def _dequant(sd: dict) -> dict:
    """Weights stored as int8 with one float16 scale per output row go back to float16."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".__q"):
            out[k[:-4]] = (v.float() * sd[k[:-4] + ".__s"].float()).half()
        elif not k.endswith(".__s"):
            out[k] = v
    return out


def _load(path: str, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)).transpose(2, 0, 1)


def _int8(net: torch.nn.Module) -> torch.nn.Module:
    """float32 then int8 dynamic quantisation of the linear layers, one child at a time."""
    for name, child in list(net.named_children()):
        if isinstance(child, torch.nn.Sequential):
            for i in range(len(child)):
                child[i] = torch.ao.quantization.quantize_dynamic(child[i].float(), {torch.nn.Linear}, dtype=torch.qint8)
        else:
            q = child.float()
            if any(isinstance(x, torch.nn.Linear) for x in q.modules()):
                q = torch.ao.quantization.quantize_dynamic(q, {torch.nn.Linear}, dtype=torch.qint8)
            setattr(net, name, q)
    for p in net.parameters(recurse=False):
        p.data = p.data.float()
    return net


def gallery_logits(gallery: list[dict]) -> torch.Tensor:
    return _gallery_logits(gallery, _NETS, _CFG)


@torch.no_grad()
def _gallery_logits(gallery: list[dict], nets: list, cfg: dict) -> torch.Tensor:
    h, w = cfg["h"], cfg["w"]
    use_flip = bool(cfg.get("flip", True))
    out = []
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(gallery), BATCH_SIZE):
            x = np.stack(list(pool.map(lambda g: _load(g["image_path"], w, h), gallery[i:i + BATCH_SIZE])))
            xr = torch.from_numpy(x).to(_DEV)
            z = 0
            for net, norm, size in nets:
                xb = norm(xr if size is None else
                          F.interpolate(xr.float(), size=size, mode="bilinear", antialias=True, align_corners=False))
                with torch.autocast(_DEV, dtype=torch.bfloat16, enabled=_DEV == "cuda"):
                    zb = net(xb).float()
                    if use_flip:
                        zb = (zb + net(xb.flip(3)).float()) / 2
                    z = z + zb
            out.append((z / len(nets)).float())
    return torch.cat(out)


def rank_gallery(sample: dict) -> dict:
    load_model()
    names = _CFG["attribute_names"]
    order = [names.index(n) for n in sample["attribute_names"]]
    z = gallery_logits(sample["gallery"])
    if _CFG.get("prior_shift"):
        # undo the training-set attribute prior: z - lam * (1 - 2 r)
        z = z - torch.as_tensor(_CFG["prior_shift"], dtype=z.dtype, device=z.device)
    z = z[:, order]
    q = torch.as_tensor(np.asarray(sample["queries"], dtype=np.float32), device=_DEV)
    sims = [loglik_scores(q[i:i + 256], z, _CFG.get("temp", 1.0), _CFG.get("alpha", 0.0),
                          _CFG.get("group_weight", 0.0)).cpu()
            for i in range(0, len(q), 256)]
    s = torch.cat(sims).numpy().astype(np.float32)
    beta = _CFG.get("query_norm", 0.0)
    if beta:
        # each gallery image matches one query: subtract beta * logsumexp over queries,
        # column blocks so the full matrix is never copied
        for j in range(0, s.shape[1], 4096):
            blk = torch.from_numpy(s[:, j:j + 4096])
            s[:, j:j + 4096] -= beta * torch.logsumexp(blk, 0, keepdim=True).numpy()
    print(f"similarities {s.shape}", flush=True)
    return {"similarities": s}
