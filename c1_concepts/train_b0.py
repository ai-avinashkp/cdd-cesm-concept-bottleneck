"""C1 image baseline (B0 = v2.1 targets; B1+ = v2.2 targets with exclusions): image -> concepts (+ label), 5 patient-grouped folds, EfficientNet-B0 at 512x768, 8 GB GPU.

Each image (LE or DES, CC or MLO) is one sample. Heads:
  concepts: 36 logits (16 LE + 20 DES); BCE only on the concepts of the image's own energy, with targets from
            that image's own report (extractor v2.1)
  label:    3 logits (Normal/Benign/Malignant), CE on the image-level label
Safe shutdown and resume: a full checkpoint (model, optimizer, schedule, AMP scaler, RNG) is saved atomically
after every epoch. Create a file named STOP in this folder to stop after the current epoch, or press Ctrl+C
(the unfinished epoch is discarded). Rerunning the same command resumes from the last checkpoint; finished
folds are skipped. Each finished fold keeps oof_fold<k>.csv and model_fold<k>.pt.

Usage: python -u train_b0.py [--epochs 12] [--folds 0,1,2,3,4]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from concepts import DES, LE  # noqa: E402

TAB = HERE.parent / "pipeline" / "tables"
CACHE = HERE / "cache_512x768"
OUT = HERE / "runs" / "b1"   # set by --run
W, H = 512, 768
CONCEPTS = [f"le_{c}" for c in LE] + [f"des_{c}" for c in DES]
CLASSES = ["Normal", "Benign", "Malignant"]


def build_cache(img):
    CACHE.mkdir(exist_ok=True)
    for name, path, side in zip(img.Image_name, img.path, img.Side):
        dst = CACHE / f"{name}.png"
        if dst.exists():
            continue
        im = Image.open(path).convert("L")
        if side == "R":                       # align all breasts to the same chest-wall side
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.resize((W, H), Image.BILINEAR).save(dst)


def image_table():
    """Targets and exclusion mask from build_concept_targets.py (v2.2): mask = 0 for the other energy's
    concepts and for excluded (conflicting) cells, so those are never trained on."""
    img = pd.read_csv(TAB / "images.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient")
    tgt = pd.read_csv(HERE / "concepts_image.csv").set_index("Image_name").loc[img.Image_name, CONCEPTS]
    mask = pd.read_csv(HERE / "concept_mask_image.csv").set_index("Image_name").loc[img.Image_name, CONCEPTS]
    img["y"] = img.label.map({c: i for i, c in enumerate(CLASSES)})
    return img, tgt.to_numpy(np.float32), mask.to_numpy(np.float32)


class DS(Dataset):
    def __init__(self, df, tgt, mask, train):
        self.df, self.tgt, self.mask = df.reset_index(drop=True), tgt, mask
        aug = [T.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08)),
               T.ColorJitter(brightness=0.15, contrast=0.15)] if train else []
        self.tf = T.Compose([T.ToImage(), *aug, T.ToDtype(torch.float32, scale=True),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(CACHE / f"{r.Image_name}.png").convert("RGB")
        return self.tf(im), torch.tensor(self.tgt[i]), torch.tensor(self.mask[i]), int(r.y)


def make_model():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(CONCEPTS) + len(CLASSES)))
    return nn.Sequential(m, head)


def run_fold(k, img, tgt, mask, epochs, dev):
    done = OUT / f"oof_fold{k}.csv"
    if done.exists():
        print(f"fold {k}: already done, skipping", flush=True)
        return True
    tr, te = (img.fold != k).to_numpy(), (img.fold == k).to_numpy()
    dl_tr = DataLoader(DS(img[tr], tgt[tr], mask[tr], True), batch_size=12, shuffle=True, num_workers=4,
                       persistent_workers=True, drop_last=True)
    dl_te = DataLoader(DS(img[te], tgt[te], mask[te], False), batch_size=24, num_workers=4)
    model = make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl_tr), pct_start=0.15)
    scaler = torch.amp.GradScaler()
    bce, ce = nn.BCEWithLogitsLoss(reduction="none"), nn.CrossEntropyLoss()
    nc = len(CONCEPTS)

    # checkpoint after every epoch (model, optimizer, schedule, AMP scaler, RNG); resume from the exact epoch
    ckpt = OUT / f"ckpt_fold{k}.pt"
    start = 0
    if ckpt.exists():
        s = torch.load(ckpt, map_location="cpu", weights_only=False)   # RNG states must stay CPU ByteTensors
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        torch.set_rng_state(s["rng_cpu"])
        torch.cuda.set_rng_state(s["rng_cuda"])
        np.random.set_state(s["rng_np"])
        start = s["epoch"]
        print(f"fold {k}: resumed from checkpoint after epoch {start}", flush=True)

    def save(ep_done):
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(), "rng_np": np.random.get_state(),
                    "epoch": ep_done}, tmp)
        tmp.replace(ckpt)                      # atomic: a crash mid-save never corrupts the last checkpoint

    for ep in range(start, epochs):
        model.train()
        t0, tot = time.monotonic(), 0.0
        try:
            for x, yc, mc, yl in dl_tr:
                x, yc, mc, yl = x.to(dev), yc.to(dev), mc.to(dev), yl.to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                    lc = (bce(out[:, :nc].float(), yc) * mc).sum() / mc.sum()
                    loss = lc + ce(out[:, nc:].float(), yl)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
        except KeyboardInterrupt:
            print(f"Ctrl+C: stopped inside epoch {ep + 1}; resume restarts it from the checkpoint after epoch {ep}",
                  flush=True)
            return False
        save(ep + 1)
        print(f"fold {k} epoch {ep + 1}/{epochs} loss {tot / len(dl_tr):.4f} ({time.monotonic() - t0:.0f}s) "
              f"[checkpoint saved]", flush=True)
        if (HERE / "STOP").exists():
            print(f"STOP file found: stopped safely after epoch {ep + 1} (checkpoint saved; rerun the same "
                  f"command to resume)", flush=True)
            return False
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, *_ in dl_te:
            o = model(x.to(dev)).float()
            preds.append(torch.cat([torch.sigmoid(o[:, :nc]), torch.softmax(o[:, nc:], 1)], 1).cpu().numpy())
    p = np.concatenate(preds)
    df = pd.DataFrame(p, columns=[f"p_{c}" for c in CONCEPTS] + [f"p_{c}" for c in CLASSES])
    df.insert(0, "Image_name", img.Image_name[te].to_numpy())
    torch.save(model.state_dict(), OUT / f"model_fold{k}.pt")    # final weights kept for later analyses
    df.to_csv(done, index=False)
    ckpt.unlink(missing_ok=True)                                  # fold finished: training state not needed
    print(f"fold {k}: wrote {done.name} and model_fold{k}.pt", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--run", default="b1")
    a = ap.parse_args()
    global OUT
    OUT = HERE / "runs" / a.run
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    img, tgt, mask = image_table()
    t0 = time.monotonic()
    build_cache(img)
    print(f"cache ready ({time.monotonic() - t0:.0f}s); images {len(img)}; concepts {len(CONCEPTS)}", flush=True)
    dev = torch.device("cuda")
    for k in [int(f) for f in a.folds.split(",")]:
        if not run_fold(k, img, tgt, mask, a.epochs, dev):
            break
    print(f"total {(time.monotonic() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
