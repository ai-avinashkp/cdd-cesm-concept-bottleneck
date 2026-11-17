"""Idea A step 1: extract image embeddings (pooled backbone features) from each fold model of a finished run,
for every image (all 2,006). Fold k's features come from model_fold<k>.pt; the breast-level fusion head for fold
k is then trained on the other folds' breasts and tested on fold k's breasts.
Output: runs/<run>/feats_fold<k>.npy (float16, n_images x dim) and runs/<run>/feats_index.csv (Image_name order).
Light GPU inference (~2-3 min per fold model on AC power).

Usage: python extract_features.py --run b2_mammoclip_640x1024 [--device cuda]
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_backbone as tb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    args = ast.literal_eval((run / "args.txt").read_text(encoding="utf-8"))
    W, H = args["width"], args["height"]
    img = pd.read_csv(tb.TAB / "images.csv")
    img[["Image_name"]].to_csv(run / "feats_index.csv", index=False)
    cache = HERE / f"cache_{W}x{H}"
    dev = torch.device(a.device)
    for k in range(5):
        out = run / f"feats_fold{k}.npy"
        if out.exists():
            print(f"fold {k}: exists, skip", flush=True)
            continue
        model = tb.Net(args["backbone"])
        model.load_state_dict(torch.load(run / f"model_fold{k}.pt", map_location="cpu", weights_only=True))
        model = model.to(dev).eval()
        mean = torch.tensor(model.backbone.norm[0])[:, None, None]
        std = torch.tensor(model.backbone.norm[1])[:, None, None]
        feats = []
        with torch.no_grad(), torch.autocast(a.device if a.device == "cuda" else "cpu", dtype=torch.float16,
                                             enabled=a.device == "cuda"):
            for i in range(0, len(img), a.batch):
                xs = []
                for n in img.Image_name[i:i + a.batch]:
                    im = np.asarray(Image.open(cache / f"{n}.png").convert("RGB"), np.float32) / 255.0
                    xs.append((torch.from_numpy(im).permute(2, 0, 1) - mean) / std)
                f = model.backbone(torch.stack(xs).to(dev)).float().cpu().numpy()
                feats.append(f.astype(np.float16))
        np.save(out, np.concatenate(feats))
        print(f"fold {k}: {out.name} {np.concatenate(feats).shape}", flush=True)


if __name__ == "__main__":
    main()
