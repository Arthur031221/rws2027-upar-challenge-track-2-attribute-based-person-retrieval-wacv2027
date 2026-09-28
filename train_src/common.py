"""Shared pieces for UPAR Track 1: the metric, attribute groups and post-processing.

The evaluation image is pytorch 2.4.1 with nothing else, so everything the
submission imports uses only numpy, torch and torchvision.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-20

# Attributes a source never annotates (written as 0 by the harmonisation).
# The Evaluation page excludes invalid attributes per dataset; train and val
# have zero positives for exactly these colours in Market-1501.
INVALID = {"Market1501": [10, 13, 14, 26, 29]}

# Groups with at least one positive in (almost) every image: if none clears
# the cut-off, the most likely member is switched on. Hair is not exhaustive.
GROUPS = [
    [0, 1, 2],                                    # age
    list(range(8, 20)),                           # upper-body colour
    list(range(21, 33)),                          # lower-body colour
    [33, 34],                                     # lower-body type
]


def valid_mask(ds_name: str, n_attr: int = 40) -> np.ndarray:
    m = np.ones(n_attr, bool)
    m[INVALID.get(ds_name, [])] = False
    return m


def upar_metrics(gt: np.ndarray, prob: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """The Evaluation page's metrics for one dataset. gt, prob: (N, 40)."""
    if valid is not None:
        gt, prob = gt[:, valid], prob[:, valid]
    pred = prob > 0.5
    g = gt == 1
    tp = (pred & g).sum(0); tn = (~pred & ~g).sum(0)
    fp = (pred & ~g).sum(0); fn = (~pred & g).sum(0)
    ma = np.mean((tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1)) / 2)
    prec_l = tp / np.maximum(tp + fp, 1); rec_l = tp / np.maximum(tp + fn, 1)
    label_f1 = np.mean(2 * prec_l * rec_l / np.maximum(prec_l + rec_l, EPS))
    inter = (pred & g).sum(1); union = (pred | g).sum(1)
    ip = inter / np.maximum(pred.sum(1), EPS); ir = inter / np.maximum(g.sum(1), EPS)
    ia = inter / np.maximum(union, EPS)
    ip_m, ir_m = ip.mean(), ir.mean()
    if1 = 2 * ip_m * ir_m / max(ip_m + ir_m, EPS)
    avg = 2 * ma * if1 / max(ma + if1, EPS)
    return {"Challenge_Avg": avg, "mA": ma, "Label_F1": label_f1, "Instance_Acc": ia.mean(),
            "Instance_Prec": ip_m, "Instance_Rec": ir_m, "Instance_F1": if1}


def per_dataset(names: np.ndarray, gt: np.ndarray, prob: np.ndarray) -> dict:
    """Mean over datasets, as the leaderboard computes it."""
    ds = np.array([n.split("/")[0] for n in names])
    out = {d: upar_metrics(gt[ds == d], prob[ds == d], valid_mask(d, gt.shape[1])) for d in sorted(set(ds))}
    out["mean"] = {k: float(np.mean([out[d][k] for d in out])) for k in next(iter(out.values()))}
    return out


def postprocess(logits: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """sigmoid(z - b), then switch on the argmax of any empty exhaustive group."""
    p = 1 / (1 + np.exp(-(logits - offsets)))
    for g in GROUPS:
        sub = p[:, g]
        empty = (sub <= 0.5).all(1)
        if empty.any():
            rows = np.where(empty)[0]
            cols = np.array(g)[sub[rows].argmax(1)]
            p[rows, cols] = 0.51
    return np.clip(p, 1e-4, 1 - 1e-4)
