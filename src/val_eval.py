"""Local validation of attribute-query scorers on the UPAR Track 2 val split.

The attribute model was trained on the task1 train split only, so the val split
(33,407 images, 3,462 queries; same images as task1 val) is held out.

    python val_eval.py --ckpt ../../../18180/runs/cnxs_e10_ema/final.pt --arch convnext_small \
        --cache ../../../18180/data/cache_val_256x128.npy --ann ../../../18180/starter/data/annotations/task2/val \
        --out ../../runs/val_cnxs_e10_ema
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import Normalize, build
from retrieval import l1_scores, loglik_scores, retrieval_metrics


@torch.no_grad()
def logits(net, norm, x, bs=256):
    out = []
    for i in range(0, len(x), bs):
        xb = norm(torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(((net(xb).float() + net(xb.flip(3)).float()) / 2).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="convnext_small")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gt_df = pd.read_csv(Path(a.ann) / "gt.csv")
    names = gt_df.iloc[:, 0].values
    gt = torch.tensor(gt_df.iloc[:, 1:].values, dtype=torch.float32)
    zf = out / "val_logits.pt"
    if zf.exists():
        z = torch.load(zf)
    else:
        net = build(a.arch).cuda().eval()
        net.load_state_dict({k: v.float() for k, v in torch.load(a.ckpt, map_location="cuda").items()})
        z = logits(net, Normalize(a.arch).cuda(), np.load(a.cache, mmap_mode="r"))
        torch.save(z, zf)
        del net; torch.cuda.empty_cache()
    ds = np.array([n.split("/")[0] for n in names])
    # galleries: the whole val split and each source on its own (the test is one unseen domain)
    galleries = {"all": np.ones(len(ds), bool)} | {d: ds == d for d in sorted(set(ds))}
    scorers = {"l1": lambda q, zz: l1_scores(q, zz)}
    for t in (1.0, 1.5, 2.0):
        for al in (0.0, 2.0, 5.0):
            scorers[f"ll_T{t}_a{al}"] = (lambda t, al: lambda q, zz: loglik_scores(q, zz, t, al))(t, al)
    res = {}
    for gname, m in galleries.items():
        g = gt[m].cuda(); zz = z[m].cuda()
        q = torch.unique(g, dim=0)          # every distinct vector in the gallery is a query
        for sname, fn in scorers.items():
            r = retrieval_metrics(fn(q, zz), q, g)
            res[f"{gname}/{sname}"] = r
            print(f"{gname:10s} {sname:14s} Q={len(q):5d} G={int(m.sum()):6d} "
                  + " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
    json.dump(res, open(out / "val_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
