"""Train the PAR model on the organizers' train split for a fixed number of epochs (EMA weights).

    python train.py --arch convnext_small --epochs 15 --out ../runs/cnxs_e15
    python train.py ... --trainval      # final fit on train+val, fixed epochs, no selection
    python train.py ... --holdout PETA  # leave one source out: train on the other two sources'
                                        # train+val, save logits for the held-out source's train+val
    python train.py ... --teacher ../runs/vitl_tv/final.pt --teacher_arch vit_l14_clip --kd 0.5
                                        # distil a frozen teacher: BCE on its sigmoid outputs
    python train.py ... --teacher_flip  # teacher targets averaged over the view and its flip
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import copy

from common import INVALID, per_dataset
from model import Normalize, build, input_size

ROOT = Path(__file__).resolve().parents[3] / "18180"  # Track 1 data and annotations
DATA = ROOT / "data"
ANN = ROOT / "starter" / "data" / "annotations" / "task1"


def split(name, h, w):
    df = pd.read_csv(ANN / name / "gt.csv")
    x = np.load(DATA / f"cache_{name}_{h}x{w}.npy", mmap_mode="r")
    return x, df.iloc[:, 1:].values.astype(np.float32), df.iloc[:, 0].values


class Pool:
    """Rows of several memory-mapped image caches, read per batch.

    Parallel runs share the page cache instead of each holding a 12 GB copy."""

    def __init__(self, caches, part, row):
        self.caches, self.part, self.row = caches, np.asarray(part), np.asarray(row)

    def __len__(self):
        return len(self.row)

    def subset(self, mask):
        return Pool(self.caches, self.part[mask], self.row[mask])

    def __getitem__(self, idx):
        idx = np.arange(len(self))[idx] if isinstance(idx, slice) else np.asarray(idx)
        out = np.empty((len(idx),) + self.caches[0].shape[1:], np.uint8)
        for k, c in enumerate(self.caches):
            sel = self.part[idx] == k
            if sel.any():
                r = self.row[idx[sel]]; o = np.argsort(r)  # sorted reads are faster on a memmap
                out[np.flatnonzero(sel)[o]] = c[r[o]]
        return out


def augment(x: torch.Tensor, pad: int) -> torch.Tensor:
    """uint8 NCHW on GPU -> flipped, shifted uint8. No hue change: colour is a label."""
    n, _, h, w = x.shape
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x.float(), (pad,) * 4, mode="replicate")
    dy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    dx = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    iy = (dy[:, None] + torch.arange(h, device=x.device)[None]).long()
    ix = (dx[:, None] + torch.arange(w, device=x.device)[None]).long()
    out = xp[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
             iy[:, None, :, None], ix[:, None, None, :]]
    b = (1 + 0.2 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5))
    return (out * b).clamp(0, 255)


@torch.no_grad()
def lowres(x: torch.Tensor, p: float, zoom: float = 0.85) -> torch.Tensor:
    """Float NCHW -> zoomed in by 1 to 1/zoom at a random position, then with probability p
    downsampled by 0.3 to 1 and scaled back, so the model sees the low-resolution crops the
    sources contain."""
    n, _, h, w = x.shape
    s = zoom + (1 - zoom) * torch.rand(n, device=x.device)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = s; theta[:, 1, 1] = s
    theta[:, :, 2] = (1 - s)[:, None] * (2 * torch.rand(n, 2, device=x.device) - 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear",
                      padding_mode="border", align_corners=False)
    for i in torch.nonzero(torch.rand(n) < p).flatten().tolist():
        f = 0.3 + 0.7 * torch.rand(1).item()
        small = F.interpolate(x[i:i + 1], size=(max(8, round(h * f)), max(4, round(w * f))),
                              mode="bilinear", antialias=True, align_corners=False)
        x[i:i + 1] = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def resize(x: torch.Tensor, size) -> torch.Tensor:
    return x if size is None else F.interpolate(x.float(), size=size, mode="bilinear", antialias=True, align_corners=False)


def layer_groups(net, a):
    """ViT: layer-wise lr decay from the top block down; no weight decay on norms, biases, embeddings."""
    depth = len(net.blocks); groups = {}
    for n, p in net.named_parameters():
        if n.startswith("head."):
            lid, lr = depth + 1, a.lr_head
        else:
            lid = int(n.split(".")[1]) + 1 if n.startswith("blocks.") else (depth + 1 if n.startswith("norm.") else 0)
            lr = a.lr * a.layer_decay ** max(0, depth - lid)
        nowd = p.ndim == 1 or n in ("pos_embed", "cls_token")
        g = groups.setdefault((lr, nowd), {"params": [], "lr": lr, "weight_decay": 0.0 if nowd else a.wd})
        g["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def predict(net, norm, x, bs=512, tta=True, logits=False, size=None):
    """Sigmoid probabilities (or logits) averaged over the image and its flip."""
    net.eval(); outs = []
    for i in range(0, len(x), bs):
        xb = resize(torch.from_numpy(x[i:i + bs]).cuda(non_blocking=True), size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = net(norm(xb)).float()
            if tta:
                lo = (lo + net(norm(xb.flip(3))).float()) / 2
        outs.append((lo if logits else torch.sigmoid(lo)).cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_small")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--loss", default="wbce", choices=["bce", "wbce"])
    ap.add_argument("--ls", type=float, default=0.05, help="label smoothing")
    ap.add_argument("--ema", type=float, default=0.9998)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainval", action="store_true")
    ap.add_argument("--holdout", choices=["Market1501", "PETA", "PA100k"])
    ap.add_argument("--lr_head", type=float, default=5e-4, help="ViT head lr")
    ap.add_argument("--layer_decay", type=float, default=0.8)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--teacher", help="frozen teacher checkpoint (final.pt)")
    ap.add_argument("--teacher_arch", default="vit_l14_clip")
    ap.add_argument("--kd", type=float, default=0.5, help="weight of the distillation loss")
    ap.add_argument("--teacher_flip", action="store_true", help="teacher targets averaged over the view and its flip")
    ap.add_argument("--lowres", type=float, default=0.0, help="probability of the low-resolution augmentation")
    ap.add_argument("--src_balance", action="store_true",
                    help="weight each image's loss by N/(K*N_source) over the training sources")
    ap.add_argument("--head_init", help="ViT head from text_head.py output instead of zeros")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    ctr, ytr, ntr = split("train", a.h, a.w)
    cva, yva, nva = split("val", a.h, a.w)
    caches = [ctr, cva]
    xtr = Pool(caches, np.zeros(len(ctr), int), np.arange(len(ctr)))
    xva = Pool(caches, np.ones(len(cva), int), np.arange(len(cva)))
    if a.trainval or a.holdout:
        xtr = Pool(caches, np.r_[xtr.part, xva.part], np.r_[xtr.row, xva.row])
        ytr = np.concatenate([ytr, yva]); ntr = np.concatenate([ntr, nva])
    if a.holdout:
        ho = np.array([n.split("/")[0] == a.holdout for n in ntr])
        xva, yva, nva = xtr.subset(ho), ytr[ho], ntr[ho]
        xtr, ytr, ntr = xtr.subset(~ho), ytr[~ho], ntr[~ho]
    size = input_size(a.arch); vit = size is not None
    rate = torch.tensor(ytr.mean(0)).cuda()
    wpos, wneg = torch.exp(1 - rate), torch.exp(rate)
    # Loss mask: attributes a source never annotates carry no label.
    lmask = np.ones_like(ytr)
    src = np.array([n.split("/")[0] for n in ntr])
    for d, cols in INVALID.items():
        lmask[np.ix_(src == d, cols)] = 0
    if a.src_balance:  # every source carries the same total weight; mean image weight stays 1
        names, counts = np.unique(src, return_counts=True)
        per = dict(zip(names, len(src) / (len(names) * counts)))
        lmask = lmask * np.array([per[s] for s in src], dtype=lmask.dtype)[:, None]
    mt = torch.from_numpy(lmask)

    net = build(a.arch, pretrained=True, drop_path=a.drop_path).cuda()
    if vit:  # start from the label prior, not from noise
        nn_ = net.head; nn_.weight.data.zero_()
        nn_.bias.data.copy_(torch.logit(torch.from_numpy(ytr.mean(0)).clamp(1e-3, 1 - 1e-3)))
        if a.head_init:  # text directions; the mean logit shift is taken out of the bias
            hi = torch.load(a.head_init, map_location="cpu")
            if a.holdout and hi["exclude"] != a.holdout:
                raise SystemExit(f"{a.head_init} was scaled with {hi['exclude']} images, holdout is {a.holdout}")
            nn_.weight.data.copy_(hi["weight"])
            nn_.bias.data.sub_(hi["mean"].to(nn_.bias.device))
    else:
        net = net.to(memory_format=torch.channels_last)
    norm = Normalize(a.arch).cuda()
    teacher = None
    if a.teacher:
        teacher = build(a.teacher_arch)
        teacher.load_state_dict({k: v.float() if v.is_floating_point() else v
                                 for k, v in torch.load(a.teacher, map_location="cpu").items()})
        teacher = teacher.cuda().eval().requires_grad_(False)
        tnorm, tsize = Normalize(a.teacher_arch).cuda(), input_size(a.teacher_arch)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(layer_groups(net, a) if vit else net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(xtr) // a.bs); warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    xt = xtr; yt = torch.from_numpy(ytr)
    best, log, step = -1, [], 0
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); perm = torch.randperm(len(xt)); tot = 0
        for i in range(len(xt) // a.bs):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            xa = augment(torch.from_numpy(xt[idx.numpy()]).cuda(non_blocking=True), pad=8)
            if a.lowres > 0:
                xa = lowres(xa, a.lowres)
            xb = resize(xa, size)
            yb = yt[idx].cuda(non_blocking=True); mb = mt[idx].cuda(non_blocking=True)
            ys = yb * (1 - a.ls) + 0.5 * a.ls
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = net(((xb / 255 - norm.mean) / norm.std).contiguous(memory_format=torch.channels_last)).float()
            w = torch.where(yb > 0.5, wpos, wneg) if a.loss == "wbce" else torch.ones_like(yb)
            loss = F.binary_cross_entropy_with_logits(lo, ys, weight=w * mb)
            if teacher is not None:  # the teacher also labels the attributes a source leaves out
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    xt_ = tnorm(resize(xa, tsize))
                    tl = teacher(xt_).float()
                    if a.teacher_flip:
                        tl = (tl + teacher(xt_.flip(3)).float()) / 2
                    tp = torch.sigmoid(tl)
                loss = (1 - a.kd) * loss + a.kd * F.binary_cross_entropy_with_logits(lo, tp)
            opt.zero_grad(set_to_none=True); loss.backward()
            if vit:
                torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip_grad)
            opt.step(); sched.step()
            step += 1; d = min(a.ema, (1 + step) / (10 + step))  # EMA warm-up
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn, 1 - d)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / (len(xt) // a.bs), "sec": time.time() - t0}
        if not a.trainval and (not a.holdout or ep == a.epochs - 1):
            m = per_dataset(nva, yva, predict(ema, norm, xva, size=size))
            rec.update({k: m["mean"][k] for k in ["Challenge_Avg", "mA", "Instance_F1"]})
            rec["per_ds"] = {d: round(m[d]["Challenge_Avg"], 4) for d in m if d != "mean"}
            best = max(best, m["mean"]["Challenge_Avg"])
        print(json.dumps(rec), flush=True); log.append(rec)
    # Fixed epochs, no selection on in-domain val: the EMA weights at the end are the model.
    if not a.trainval:  # logits of the evaluation pool, for offset fitting
        np.savez(out / "eval_logits.npz", logits=predict(ema, norm, xva, logits=True, size=size), gt=yva, names=nva)
        # the submitted package runs without flip, so keep the matching logits for the gate
        np.savez(out / "eval_logits_noflip.npz", logits=predict(ema, norm, xva, tta=False, logits=True, size=size),
                 gt=yva, names=nva)
    # Holdout runs keep their weights too, as teachers for the same fold.
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()}
               if vit else ema.state_dict(), out / "final.pt")
    json.dump({"args": vars(a), "log": log, "best_val_avg": best}, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
