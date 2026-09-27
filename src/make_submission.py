"""Pack a Track 2 code submission from an attribute-model checkpoint.

    python make_submission.py --ckpt ../../../18180/runs/cnxs_e10_ema/final.pt --arch convnext_small \
        --out ../../runs/sub_cnxs_e10_ema
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SUB = HERE.parent / "submission"
NAMES = ["Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Hair-Length-Short", "Hair-Length-Long",
         "Hair-Length-Bald", "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
         "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange",
         "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red", "UpperBody-Color-White",
         "UpperBody-Color-Yellow", "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
         "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green", "LowerBody-Color-Grey",
         "LowerBody-Color-Orange", "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red",
         "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
         "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
         "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--arch", default="convnext_small")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    for f in ["run.py", "metadata.yaml"]:
        shutil.copy(SUB / f, out / f)
    for f in ["model.py", "vit.py", "retrieval.py"]:
        shutil.copy(HERE / f, out / f)
    models = []
    for i, c in enumerate(a.ckpt):
        sd = torch.load(c, map_location="cpu")
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, out / f"m{i}.pt")
        models.append({"arch": a.arch, "weights": f"m{i}.pt"})
    json.dump({"models": models, "h": a.h, "w": a.w, "temp": a.temp, "alpha": a.alpha,
               "attribute_names": NAMES}, open(out / "config.json", "w"), indent=1)
    with zipfile.ZipFile(str(out) + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f, f.name)
    print("wrote", str(out) + ".zip")


if __name__ == "__main__":
    main()
