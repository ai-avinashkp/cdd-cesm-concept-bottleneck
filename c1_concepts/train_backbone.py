"""C1 image model with a choice of backbone (model sweep B2+). Same task, targets, folds and outputs as
train_b0.py (B0/B1), so runs compare directly.

Each image (LE or DES, CC or MLO) is one sample. Heads: 35 concept logits (BCE on the image's own-energy
concepts, excluded cells masked; targets v2.2) + 3 label logits (CE on the image label).

Backbones:
  effb0         torchvision EfficientNet-B0, ImageNet (= B1 when run at 512x768)
  mammoclip_b2  Mammo-CLIP image encoder (EfficientNet-B2, mammography image-text pretraining; CC BY-NC-SA 4.0),
                loaded from ../weights/mammoclip/b2-model-best-epoch-10.tar
  effv2s        timm tf_efficientnetv2_s.in21k_ft_in1k (Apache 2.0)
  dinov2b       timm vit_base_patch14_dinov2.lvd142m (Apache 2.0), dynamic image size, grad checkpointing

Safe shutdown and resume: a full checkpoint (model, optimizer, schedule, AMP scaler, RNG) is saved atomically
after every epoch. Create a file named STOP in this folder to stop after the current epoch, or press Ctrl+C
(the unfinished epoch is discarded). Rerunning the same command resumes from the last checkpoint; finished
folds are skipped. Each finished fold keeps oof_fold<k>.csv and model_fold<k>.pt.

Usage: python -u train_backbone.py --backbone mammoclip_b2 --width 640 --height 1024 --run b2_mammoclip
       python -u train_backbone.py --backbone effv2s --bench    (time 20 training steps, no training)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from concepts import DES, LE  # noqa: E402

TAB = HERE.parent / "pipeline" / "tables"
WEIGHTS = HERE.parent / "weights"
CONCEPTS = [f"le_{c}" for c in LE] + [f"des_{c}" for c in DES]
CLASSES = ["Normal", "Benign", "Malignant"]
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
GRAD_CKPT = True          # ViT gradient checkpointing (less memory, slower); --no-gc turns it off


class GeM(nn.Module):
    """Generalised-mean pooling, as in the Mammo-CLIP image encoder."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p, self.eps = p, eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class Backbone(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "effb0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            m.classifier = nn.Identity()
            self.net, self.dim, self.chans, self.norm = m, 1280, 3, IMAGENET
        elif name == "mammoclip_b2":
            self.net, self.dim, self.chans, self.norm = load_mammoclip("b2-model-best-epoch-10.tar")
            self.pool = GeM()
        elif name == "effv2s":
            m = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0)
            cfg = m.pretrained_cfg
            self.net, self.dim, self.chans, self.norm = m, m.num_features, 3, (list(cfg["mean"]), list(cfg["std"]))
        elif name == "dinov2b":
            m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                  dynamic_img_size=True)
            m.set_grad_checkpointing(GRAD_CKPT)
            self.net, self.dim, self.chans, self.norm = m, m.num_features, 3, IMAGENET
        else:
            raise ValueError(name)

    def forward(self, x):
        if self.name == "mammoclip_b2":
            return self.pool(self.net.extract_features(x))
        return self.net(x)


def load_mammoclip(fname):
    """Mammo-CLIP image encoder = EfficientNet-B2 in efficientnet_pytorch layout (keys image_encoder._conv_stem,
    _blocks.0-22, _conv_head 1408), 3 input channels, grayscale normalised with mean 0.3089 / std 0.2505
    (checkpoint config 'base'), native size 1520x912. The downloaded pickle is opened with a restricted
    unpickler (safe_ckpt.py) that only resolves allowlisted classes."""
    from efficientnet_pytorch import EfficientNet
    from safe_ckpt import load_restricted
    ck = load_restricted(WEIGHTS / "mammoclip" / fname)
    enc = {k[len("image_encoder."):]: v for k, v in ck["model"].items() if k.startswith("image_encoder.")}
    m = EfficientNet.from_name("efficientnet-b2")
    missing, unexpected = m.load_state_dict(enc, strict=False)
    missing = [k for k in missing if not k.startswith("_fc")]
    if missing or unexpected:
        raise SystemExit(f"Mammo-CLIP load mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    base = ck["config"]["base"]
    mean, std = float(base["mean"]), float(base["std"])
    print(f"Mammo-CLIP encoder: efficientnet-b2, {len(enc)} tensors loaded, norm ({mean:.4f}, {std:.4f})", flush=True)
    return m, 1408, 3, ([mean] * 3, [std] * 3)


def build_cache(img, w, h):
    cache = HERE / f"cache_{w}x{h}"
    cache.mkdir(exist_ok=True)
    for name, path, side in zip(img.Image_name, img.path, img.Side):
        dst = cache / f"{name}.png"
        if dst.exists():
            continue
        im = Image.open(path).convert("L")
        if side == "R":                       # align all breasts to the same chest-wall side
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.resize((w, h), Image.BILINEAR).save(dst)
    return cache


def image_table():
    img = pd.read_csv(TAB / "images.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient")
    tgt = pd.read_csv(HERE / "concepts_image.csv").set_index("Image_name").loc[img.Image_name, CONCEPTS]
    mask = pd.read_csv(HERE / "concept_mask_image.csv").set_index("Image_name").loc[img.Image_name, CONCEPTS]
    img["y"] = img.label.map({c: i for i, c in enumerate(CLASSES)})
    return img, tgt.to_numpy(np.float32), mask.to_numpy(np.float32)


class DS(Dataset):
    def __init__(self, df, tgt, mask, train, cache, chans, norm):
        self.df, self.tgt, self.mask, self.cache, self.chans = df.reset_index(drop=True), tgt, mask, cache, chans
        aug = [T.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08)),
               T.ColorJitter(brightness=0.15, contrast=0.15)] if train else []
        self.tf = T.Compose([T.ToImage(), *aug, T.ToDtype(torch.float32, scale=True), T.Normalize(*norm)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(self.cache / f"{r.Image_name}.png").convert("L" if self.chans == 1 else "RGB")
        return self.tf(im), torch.tensor(self.tgt[i]), torch.tensor(self.mask[i]), int(r.y)


class Net(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = Backbone(backbone)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.backbone.dim, len(CONCEPTS) + len(CLASSES)))

    def forward(self, x):
        return self.head(self.backbone(x))


def run_fold(k, img, tgt, mask, a, out, cache, dev, bench=False):
    done = out / f"oof_fold{k}.csv"
    if done.exists() and not bench:
        print(f"fold {k}: already done, skipping", flush=True)
        return True
    model = Net(a.backbone).to(dev)
    chans, norm = model.backbone.chans, model.backbone.norm
    tr, te = (img.fold != k).to_numpy(), (img.fold == k).to_numpy()
    dl_tr = DataLoader(DS(img[tr], tgt[tr], mask[tr], True, cache, chans, norm), batch_size=a.batch, shuffle=True,
                       num_workers=4, persistent_workers=True, drop_last=True)
    dl_te = DataLoader(DS(img[te], tgt[te], mask[te], False, cache, chans, norm), batch_size=a.batch * 2,
                       num_workers=4)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.lr * 10}], weight_decay=1e-4)
    steps = a.epochs * (len(dl_tr) // a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 10], total_steps=steps, pct_start=a.warmup)
    amp_dtype = torch.bfloat16 if a.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(a.amp == "fp16"))   # bf16 needs no loss scaling
    bce, ce = nn.BCEWithLogitsLoss(reduction="none"), nn.CrossEntropyLoss()
    nc = len(CONCEPTS)

    def step_loss(x, yc, mc, yl):
        with torch.autocast("cuda", dtype=amp_dtype):
            o = model(x)
            lc = (bce(o[:, :nc].float(), yc) * mc).sum() / mc.sum().clamp(min=1)
            return a.concept_weight * lc + ce(o[:, nc:].float(), yl)   # concept_weight 0 = label-only black box

    if bench:                                   # time 20 optimizer-free training steps
        model.train()
        torch.cuda.reset_peak_memory_stats()
        it = iter(dl_tr)
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize()
                t0 = time.monotonic()
            x, yc, mc, yl = [t.to(dev) for t in next(it)]
            scaler.scale(step_loss(x, yc, mc, yl)).backward()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        per_img = (time.monotonic() - t0) / (20 * a.batch)
        n_tr = int(tr.sum())
        ep = per_img * n_tr * 1.15                # + evaluation and loader overhead
        print(f"BENCH {a.backbone} {a.width}x{a.height} batch {a.batch}: {per_img * 1000:.0f} ms/image, "
              f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, ~{ep:.0f} s/epoch, "
              f"~{ep * a.epochs * 5 / 3600:.1f} h for {a.epochs} epochs x 5 folds", flush=True)
        return True

    ckpt = out / f"ckpt_fold{k}.pt"
    start = 0
    if ckpt.exists():
        s = torch.load(ckpt, map_location="cpu", weights_only=False)   # our own file; RNG states stay on CPU
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

    for ep in range(start, epochs := a.epochs):
        model.train()
        frozen = ep < a.freeze_epochs                  # LP-FT: heads only first, then full fine-tuning
        for prm in model.backbone.parameters():
            prm.requires_grad = not frozen
        if a.freeze_epochs and ep in (0, a.freeze_epochs):
            print(f"fold {k} epoch {ep + 1}: backbone {'FROZEN (heads only)' if frozen else 'unfrozen'}", flush=True)
        t0, tot, n = time.monotonic(), 0.0, 0
        try:
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(dl_tr):
                x, yc, mc, yl = [t.to(dev) for t in batch]
                loss = step_loss(x, yc, mc, yl) / a.accum
                scaler.scale(loss).backward()
                if (i + 1) % a.accum == 0:
                    if a.clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                tot, n = tot + loss.item() * a.accum, n + 1
        except KeyboardInterrupt:
            print(f"Ctrl+C: stopped inside epoch {ep + 1}; resume restarts it from the checkpoint after epoch {ep}",
                  flush=True)
            return False
        except RuntimeError as e:          # shared node: another job's cleanup can delete DataLoader shm files
            if "shared memory" in str(e) or "DataLoader worker" in str(e):
                print(f"DataLoader error in epoch {ep + 1} ({str(e)[:80]}); reloading the checkpoint after "
                      f"epoch {ep} and redoing the epoch", flush=True)
                return "retry"
            raise
        save(ep + 1)
        print(f"fold {k} epoch {ep + 1}/{epochs} loss {tot / n:.4f} ({time.monotonic() - t0:.0f}s) "
              f"[checkpoint saved]", flush=True)
        if (HERE / "STOP").exists():
            print(f"STOP file found: stopped safely after epoch {ep + 1} (checkpoint saved; rerun the same "
                  f"command to resume)", flush=True)
            return False
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
        for x, *_ in dl_te:
            o = model(x.to(dev)).float()
            preds.append(torch.cat([torch.sigmoid(o[:, :nc]), torch.softmax(o[:, nc:], 1)], 1).cpu().numpy())
    p = np.concatenate(preds)
    df = pd.DataFrame(p, columns=[f"p_{c}" for c in CONCEPTS] + [f"p_{c}" for c in CLASSES])
    df.insert(0, "Image_name", img.Image_name[te].to_numpy())
    torch.save(model.state_dict(), out / f"model_fold{k}.pt")
    df.to_csv(done, index=False)
    ckpt.unlink(missing_ok=True)
    print(f"fold {k}: wrote {done.name} and model_fold{k}.pt", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["effb0", "mammoclip_b2", "effv2s", "dinov2b"])
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone LR (head uses 10x)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--run", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--no-gc", action="store_true", help="disable ViT gradient checkpointing (faster, more memory)")
    ap.add_argument("--seed", type=int, default=0, help="training seed (init of head, data order, augmentation)")
    ap.add_argument("--amp", choices=["fp16", "bf16"], default="fp16", help="autocast dtype (bf16: more stable ViT)")
    ap.add_argument("--warmup", type=float, default=0.15, help="OneCycle warm-up fraction")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clipping (0 = off)")
    ap.add_argument("--freeze-epochs", type=int, default=0, help="train heads only for the first N epochs (LP-FT)")
    ap.add_argument("--task", choices=["3class", "binary"], default="3class",
                    help="binary = Normal vs Tumorous (benign + malignant merged), for the G2b comparison")
    ap.add_argument("--energy", choices=["all", "LE", "DES"], default="all", help="restrict to one energy")
    ap.add_argument("--concept-weight", type=float, default=1.0,
                    help="weight of the concept loss; 0 trains a label-only black box (no concept supervision)")
    ap.add_argument("--cross-energy", action="store_true",
                    help="with --energy LE: also train each LE image on its breast's DES (contrast) concepts "
                         "(breast-level report labels, excluded cells masked): cross-energy concept bottleneck")
    ap.add_argument("--label-source", choices=["image", "breast"], default="image",
                    help="breast = every image learns its breast's label (most severe image label) instead of its "
                         "own image label; the check for the audit's aggregation finding (2026-11-17)")
    ap.add_argument("--split", choices=["patient", "image"], default="patient",
                    help="patient = shared patient-grouped folds (correct); image = random image-level folds that "
                         "let one patient's images fall in train and test (LEAKAGE experiment only)")
    a = ap.parse_args()
    global GRAD_CKPT
    GRAD_CKPT = not a.no_gc
    out = HERE / "runs" / (a.run or f"{a.backbone}_{a.width}x{a.height}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.txt").write_text(str(vars(a)), encoding="utf-8")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    img, tgt, mask = image_table()
    if a.label_source == "breast":
        bl = pd.read_csv(TAB / "breasts.csv").set_index("breast").label
        changed = int((img.label != img.breast.map(bl)).sum())
        img["y"] = img.breast.map(bl).map({"Normal": 0, "Benign": 1, "Malignant": 2})
        print(f"label source: breast labels ({changed} of {len(img)} image labels replaced)", flush=True)
    if a.task == "binary":
        global CLASSES
        CLASSES = ["Normal", "Tumorous"]
        img["y"] = (img.label != "Normal").astype(int)
    if a.energy != "all":
        keep = (img.energy == a.energy).to_numpy()
        img, tgt, mask = img[keep].reset_index(drop=True), tgt[keep], mask[keep]
        print(f"energy filter {a.energy}: {len(img)} images", flush=True)
    if a.cross_energy:
        if a.energy != "LE":
            raise SystemExit("--cross-energy needs --energy LE")
        bt = pd.read_csv(HERE / "concepts_breast.csv").set_index("breast")
        bm = pd.read_csv(HERE / "concept_mask_breast.csv").set_index("breast")
        des = [j for j, c in enumerate(CONCEPTS) if c.startswith("des_")]
        tgt[:, des] = bt.loc[img.breast, [CONCEPTS[j] for j in des]].to_numpy(np.float32)
        mask[:, des] = bm.loc[img.breast, [CONCEPTS[j] for j in des]].to_numpy(np.float32)
        print(f"cross-energy targets: LE images now also learn {len(des)} DES concepts "
              f"({int(mask[:, des].sum())} usable cells)", flush=True)
    if a.split == "image":
        from sklearn.model_selection import StratifiedKFold
        img["fold"] = -1
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(img, img.y)):
            img.loc[img.index[te], "fold"] = k
        img[["Image_name", "patient", "fold"]].to_csv(out / "folds_image_split.csv", index=False)
        shared = [len(set(img.patient[img.fold == k]) & set(img.patient[img.fold != k])) /
                  img.patient[img.fold == k].nunique() for k in range(5)]
        print(f"IMAGE-LEVEL SPLIT (leakage experiment): test patients also in train per fold "
              f"{[f'{s:.0%}' for s in shared]}", flush=True)
    t0 = time.monotonic()
    cache = build_cache(img, a.width, a.height)
    print(f"cache {cache.name} ready ({time.monotonic() - t0:.0f}s); images {len(img)}; concepts {len(CONCEPTS)}",
          flush=True)
    dev = torch.device("cuda")
    for k in [int(f) for f in a.folds.split(",")]:
        for _ in range(20):                    # "retry" = transient DataLoader failure; resume from checkpoint
            ok = run_fold(k, img, tgt, mask, a, out, cache, dev, bench=a.bench)
            if ok != "retry":
                break
        if ok is not True:
            break
        if a.bench:
            break
    print(f"total {(time.monotonic() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
