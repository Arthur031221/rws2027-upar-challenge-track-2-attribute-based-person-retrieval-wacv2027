"""UPAR Track 2 entry point, imported by the challenge ingestion program.

Two steps: predict attribute logits for every gallery image (image and its
horizontal flip averaged), then score each query by the Bernoulli
log-likelihood of its 40-bit attribute vector under the predicted
probabilities. Only torch, torchvision, numpy and Pillow are used.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from model import Normalize, build
from retrieval import loglik_scores

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 128

_NETS: list = []
_CFG: dict = {}
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> None:
    global _CFG
    if _NETS:
        return
    _CFG = json.load(open(os.path.join(HERE, "config.json")))
    for m in _CFG["models"]:
        net = build(m["arch"])
        sd = torch.load(os.path.join(HERE, m["weights"]), map_location="cpu")
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        _NETS.append((net.to(_DEV).eval(), Normalize(m["arch"]).to(_DEV)))


def _load(path: str, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)).transpose(2, 0, 1)


@torch.no_grad()
def gallery_logits(gallery: list[dict]) -> torch.Tensor:
    h, w = _CFG["h"], _CFG["w"]
    out, t0 = [], time.time()
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(gallery), BATCH_SIZE):
            x = np.stack(list(pool.map(lambda g: _load(g["image_path"], w, h), gallery[i:i + BATCH_SIZE])))
            xr = torch.from_numpy(x).to(_DEV)
            z = 0
            for net, norm in _NETS:
                xb = norm(xr)
                with torch.autocast(_DEV, dtype=torch.bfloat16, enabled=_DEV == "cuda"):
                    z = z + (net(xb).float() + net(xb.flip(3)).float()) / 2
            out.append((z / len(_NETS)).float())
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
