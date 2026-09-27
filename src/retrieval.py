"""Attribute-query scoring and the UPAR Track 2 retrieval metrics.

Everything here needs only numpy and torch, so the submission can import it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def loglik_scores(q: torch.Tensor, z: torch.Tensor, temp: float = 1.0, alpha: float = 0.0) -> torch.Tensor:
    """Similarity of every query to every gallery image, (Q, G).

    q: (Q, 40) binary queries, z: (G, 40) attribute logits.
    Bernoulli log-likelihood of the query vector under the predicted
    probabilities sigmoid(z / temp), plus alpha times the expected number of
    agreeing attributes (the degree of match that mADM credits for near misses).
    """
    q = q.float(); z = z.float() / temp
    lp, ln = F.logsigmoid(z), F.logsigmoid(-z)
    s = q @ lp.T + (1 - q) @ ln.T
    if alpha:
        p = torch.sigmoid(z)
        s = s + alpha * (q @ p.T + (1 - q) @ (1 - p).T)
    return s


def l1_scores(q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Expected attribute agreement (the organisers' L1 baseline, as a similarity)."""
    p = torch.sigmoid(z.float()); q = q.float()
    return q @ p.T + (1 - q) @ (1 - p).T


@torch.no_grad()
def retrieval_metrics(sim: torch.Tensor, q: torch.Tensor, gt: torch.Tensor, chunk: int = 128) -> dict[str, float]:
    """mADM, mAP, Rank-1/5/10 and mINP as defined on the Evaluation page.

    sim: (Q, G) similarities (larger first; ties keep gallery order),
    q: (Q, 40) queries, gt: (G, 40) gallery labels. A match is an exact 40-bit equality.
    """
    q = q.float(); gt = gt.float()
    nq, ng = sim.shape
    ar = torch.arange(1, ng + 1, device=sim.device, dtype=torch.float64)
    acc = {k: 0.0 for k in ["mADM", "mAP", "Rank-1", "Rank-5", "Rank-10", "mINP"]}
    for i in range(0, nq, chunk):
        qs = q[i:i + chunk]
        agree = (qs @ gt.T + (1 - qs) @ (1 - gt).T) / q.shape[1]          # dom, (c, G)
        match = agree > 1 - 1e-6
        mean, mx = agree.mean(1, keepdim=True), agree.max(1, keepdim=True).values
        ndom = ((agree - mean) / (mx - mean).clamp_min(1e-12)).clamp_min(0)
        # stable sort so equal scores keep gallery order
        order = torch.sort(sim[i:i + chunk], dim=1, descending=True, stable=True).indices
        m = torch.gather(match, 1, order).double()
        cd = torch.gather(ndom, 1, order).double().cumsum(1)
        R = m.sum(1)
        cm = m.cumsum(1)
        acc["mAP"] += ((cm / ar) * m).sum(1).div(R).sum().item()
        acc["mADM"] += ((cd / ar) * m).sum(1).div(R).sum().item()
        first = m.argmax(1) + 1
        last = ng - m.flip(1).argmax(1)
        acc["mINP"] += (R / last).sum().item()
        for k in (1, 5, 10):
            acc[f"Rank-{k}"] += (first <= k).sum().item()
    return {k: v / nq for k, v in acc.items()}
