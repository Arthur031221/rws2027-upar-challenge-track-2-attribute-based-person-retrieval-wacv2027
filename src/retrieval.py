"""Attribute-query scoring and the UPAR Track 2 retrieval metrics.

Everything here needs only numpy and torch, so the submission can import it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# Mutually exclusive attribute groups in the 40-attribute layout: age, hair
# length, upper colour, lower colour, lower-body type.
GROUPS = [[0, 1, 2], [4, 5, 6], list(range(8, 20)), list(range(21, 33)), [33, 34]]


def _bernoulli(q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return q @ F.logsigmoid(z).T + (1 - q) @ F.logsigmoid(-z).T


def loglik_scores(q: torch.Tensor, z: torch.Tensor, temp: float = 1.0, alpha: float = 0.0,
                  group_weight: float = 0.0) -> torch.Tensor:
    """Similarity of every query to every gallery image, (Q, G).

    q: (Q, 40) binary queries, z: (G, 40) attribute logits.
    Bernoulli log-likelihood of the query vector under the predicted
    probabilities sigmoid(z / temp), plus alpha times the expected number of
    agreeing attributes (the degree of match that mADM credits for near misses).

    With group_weight > 0, a query with exactly one positive in a mutually
    exclusive group scores that group as a mix of the categorical
    log-probability of the positive (sigmoids renormalised over the group) and
    the independent Bernoulli terms, weighted group_weight and 1 - group_weight.
    """
    q = q.float(); z = z.float() / temp
    if group_weight:
        grouped = {j for g in GROUPS for j in g}
        solo = [j for j in range(q.shape[1]) if j not in grouped]
        s = _bernoulli(q[:, solo], z[:, solo])
        for g in GROUPS:
            qg, zg = q[:, g], z[:, g]
            b = _bernoulli(qg, zg)
            lp = F.logsigmoid(zg)
            cat = qg @ (lp - torch.logsumexp(lp, 1, keepdim=True)).T
            one = (qg.sum(1, keepdim=True) == 1).float()
            s = s + one * (group_weight * cat + (1 - group_weight) * b) + (1 - one) * b
    else:
        s = _bernoulli(q, z)
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
