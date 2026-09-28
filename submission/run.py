"""UPAR Track 2 entry point, imported by the challenge ingestion program.

Two steps: predict attribute logits for every gallery image (image and its
horizontal flip averaged), then score each query by the Bernoulli
log-likelihood of its 40-bit attribute vector under the predicted
probabilities. Only torch, torchvision, numpy and Pillow are used.

Images are resized to h x w as in training and, for a ViT, resampled to the
model's input grid the same way the training loop does. The platform allows
one hour for the whole run, so after every batch the projected finish time is
checked against BUDGET_S and, if it is over, flip averaging is dropped. If the
config names fallback models and the projection is still over budget after a
few batches without flip, the gallery restarts from the first image on the
fallback models, so every gallery image is scored by the same models.

The worker's memory cap is about 2 GB. Weights are memory-mapped into a model
built on the meta device; on a GPU they run in float32 with bfloat16 autocast,
on the CPU the linear layers are quantised to int8 at load time unless the
config sets "cpu_int8" to false.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import Normalize, build, input_size
from retrieval import loglik_scores

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 64
BUDGET_S = 2700.0  # of the 3600 s limit, leaving room for start-up and scoring
PROBE_BATCHES = 3  # batches timed before giving up on the primary models

_NETS: list = []  # (net, normalise, input size or None)
_CFG: dict = {}
_DEV = "cpu"
_MODES = ("full", "noflip")
_T0 = [0.0]


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
    _T0[0] = time.time()
    _DEV = "cuda" if _cuda_works() else "cpu"
    if _DEV == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    _CFG = json.load(open(os.path.join(HERE, "config.json")))
    _NETS.extend(_load_nets(_CFG["models"]))


def _load_nets(models: list[dict]) -> list:
    nets = []
    for m in models:
        # Weights are memory-mapped and assigned into a model built on the meta
        # device, so host memory holds about one copy of the float16 weights.
        sd = torch.load(os.path.join(HERE, m["weights"]), map_location="cpu", mmap=True, weights_only=True)
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


class _Restart(Exception):
    pass


def gallery_logits(gallery: list[dict]) -> torch.Tensor:
    try:
        return _gallery_logits(gallery, _NETS, _CFG, bool(_CFG.get("fallback")))
    except _Restart:
        fb = _CFG["fallback"]
        _NETS[:] = []  # free the primary models before loading the fallback
        _NETS.extend(_load_nets(fb["models"]))
        print("restart on the fallback models", flush=True)
        return _gallery_logits(gallery, _NETS, {**_CFG, **fb}, False)


@torch.no_grad()
def _gallery_logits(gallery: list[dict], nets: list, cfg: dict, can_restart: bool) -> torch.Tensor:
    h, w = cfg["h"], cfg["w"]
    last = len(_MODES) - 1
    mode = 0 if cfg.get("flip", True) else last
    out, t0 = [], time.time()
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(gallery), BATCH_SIZE):
            tb = time.time()
            x = np.stack(list(pool.map(lambda g: _load(g["image_path"], w, h), gallery[i:i + BATCH_SIZE])))
            xr = torch.from_numpy(x).to(_DEV)
            z = 0
            for net, norm, size in nets:
                xb = norm(xr if size is None else
                          F.interpolate(xr.float(), size=size, mode="bilinear", antialias=True, align_corners=False))
                with torch.autocast(_DEV, dtype=torch.bfloat16, enabled=_DEV == "cuda"):
                    zb = net(xb).float()
                    z = z + ((zb + net(xb.flip(3)).float()) / 2 if mode == 0 else zb)
            out.append((z / len(nets)).float())
            left = len(gallery) - i - len(x)
            per_image = (time.time() - tb) / len(x)
            over = time.time() - _T0[0] + left * per_image > BUDGET_S
            if left > 0 and mode < last and over:
                mode += 1
                print(f"step down to {_MODES[mode]} at {i + len(x)} images", flush=True)
            elif left > 0 and mode == last and over and can_restart and i // BATCH_SIZE < PROBE_BATCHES:
                raise _Restart
            if (i // BATCH_SIZE) % 50 == 0:
                print(f"gallery {i + len(x)}/{len(gallery)} {time.time() - t0:.0f}s", flush=True)
    return torch.cat(out)


def rank_gallery(sample: dict) -> dict:
    load_model()
    names = _CFG["attribute_names"]
    order = [names.index(n) for n in sample["attribute_names"]]
    z = gallery_logits(sample["gallery"])[:, order]
    q = torch.as_tensor(np.asarray(sample["queries"], dtype=np.float32), device=_DEV)
    sims = [loglik_scores(q[i:i + 256], z, _CFG.get("temp", 1.0), _CFG.get("alpha", 0.0),
                          _CFG.get("group_weight", 0.0)).cpu()
            for i in range(0, len(q), 256)]
    s = torch.cat(sims).numpy().astype(np.float32)
    print(f"similarities {s.shape}", flush=True)
    return {"similarities": s}
