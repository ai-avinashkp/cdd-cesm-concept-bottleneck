"""G3: do the model's explanations point at the lesion? Grad-CAM vs the radiologists' lesion polygons
(CDD-CESM radiology_segmentations.csv, VIA format), out-of-fold (each image explained by the fold model that
did not train on it).

For every abnormal image with at least one polygon:
  black box   Grad-CAM of the label-head logit of the image's TRUE class (benign or malignant)
  concepts    Grad-CAM of the image's top predicted lesion concept (energy-specific list LESION)
Metrics per image: pointing game (CAM maximum inside a polygon), energy inside the lesion mask (share of CAM
mass), IoU of the CAM region >= 0.5 x max with the mask. Writes runs/<run>/localisation.csv + LOCALISATION.md.

Usage: python localisation.py --run b2_mammoclip_640x1024 [--device cuda] [--limit 5]
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_backbone as tb  # noqa: E402

CDDCESM = Path(os.environ.get("CDDCESM_ROOT", HERE.parent / "CDD-CESM-Dataset_original" / "CDD-CESM-Dataset"))
SEG = CDDCESM / "data" / "radiology_segmentations.csv"
LESION = {"LE": ["le_mass", "le_calc_suspicious", "le_distortion", "le_asymmetry", "le_high_density",
                 "le_margin_spiculated", "le_shape_irregular"],
          "DES": ["des_mass_enh", "des_nme", "des_heterogeneous", "des_irregular", "des_margin_spiculated",
                  "des_rim", "des_focus_foci"]}


def polygons_mask(rows, size):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for s in rows:
        g = json.loads(s) if isinstance(s, str) and s.startswith("{") else ast.literal_eval(s)
        n = g.get("name")
        if n in ("polygon", "polyline") and len(g.get("all_points_x", [])) >= 3:
            d.polygon(list(zip(g["all_points_x"], g["all_points_y"])), fill=1)
        elif n == "circle":
            cx, cy, r = g["cx"], g["cy"], g["r"]
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)
        elif n == "ellipse":
            cx, cy, rx, ry = g["cx"], g["cy"], g["rx"], g["ry"]
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=1)
        elif n == "rect":
            d.rectangle([g["x"], g["y"], g["x"] + g["width"], g["y"] + g["height"]], fill=1)
        elif n == "point":
            d.ellipse([g["cx"] - 20, g["cy"] - 20, g["cx"] + 20, g["cy"] + 20], fill=1)
    return m


def gradcam(model, x, index):
    net = model.backbone.net
    feats = net.extract_features(x)
    feats.retain_grad()
    out = model.head(model.backbone.pool(feats))
    model.zero_grad(set_to_none=True)
    out[0, index].backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1))[0].detach().float().cpu().numpy()
    return cam, out.detach()


def metrics(cam, mask):
    cam_r = np.asarray(Image.fromarray(cam).resize(mask.shape[::-1], Image.BILINEAR))
    if cam_r.max() <= 0:
        return np.nan, np.nan, np.nan
    point = float(mask.flat[np.argmax(cam_r)] > 0)
    energy = float(cam_r[mask > 0].sum() / cam_r.sum())
    top = cam_r >= 0.5 * cam_r.max()        # standard Grad-CAM binarisation (CAMs are sparse after ReLU)
    iou = float((top & (mask > 0)).sum() / (top | (mask > 0)).sum())
    return point, energy, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b2_mammoclip_640x1024")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    args = ast.literal_eval((run / "args.txt").read_text(encoding="utf-8"))
    if args["backbone"] != "mammoclip_b2":
        raise SystemExit("Grad-CAM here is written for the Mammo-CLIP (efficientnet_pytorch) backbone")
    W, H = args["width"], args["height"]
    seg = pd.read_csv(SEG)
    seg["Image_name"] = seg["#filename"].str.replace(".jpg", "", regex=False).str.strip()
    img, _, _ = tb.image_table()
    img = img[(img.label != "Normal") & img.Image_name.isin(seg.Image_name)]
    if a.limit:
        img = img.groupby("fold").head(max(1, a.limit // 5))
    cache = HERE / f"cache_{W}x{H}"
    dev = torch.device(a.device)
    rows = []
    for k in sorted(img.fold.unique()):
        model = tb.Net("mammoclip_b2")
        model.load_state_dict(torch.load(run / f"model_fold{k}.pt", map_location="cpu", weights_only=True))
        model = model.to(dev).eval()
        mean, std = model.backbone.norm
        for _, r in img[img.fold == k].iterrows():
            im = Image.open(cache / f"{r.Image_name}.png").convert("RGB")
            x = (torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1) - torch.tensor(mean)[:, None, None]) \
                / torch.tensor(std)[:, None, None]
            x = x[None].to(dev)
            orig = Image.open(r.path)
            m = polygons_mask(seg.loc[seg.Image_name == r.Image_name, "region_shape_attributes"], orig.size)
            if r.Side == "R":
                m = m.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.asarray(m.resize((W, H), Image.NEAREST))
            if mask.sum() == 0:
                continue
            nc = len(tb.CONCEPTS)
            cam_bb, out = gradcam(model, x, nc + int(r.y))
            probs = torch.sigmoid(out[0, :nc]).cpu().numpy()
            cands = [tb.CONCEPTS.index(c) for c in LESION[r.energy]]
            top = max(cands, key=lambda j: probs[j])
            cam_c, _ = gradcam(model, x, top)
            pb, eb, ib = metrics(cam_bb, mask)
            pc, ec, ic = metrics(cam_c, mask)
            rows.append({"Image_name": r.Image_name, "fold": k, "energy": r.energy, "label": r.label,
                         "mask_frac": mask.mean(), "top_concept": tb.CONCEPTS[top], "top_concept_p": probs[top],
                         "bb_point": pb, "bb_energy": eb, "bb_iou50": ib,
                         "concept_point": pc, "concept_energy": ec, "concept_iou50": ic})
        print(f"fold {k}: {sum(1 for x in rows if x['fold'] == k)} images", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "localisation.csv", index=False)
    L = [f"# G3 localisation vs radiologist polygons ({a.run}, {len(df)} abnormal images, out-of-fold)", "",
         f"Mean lesion-mask share of the image (random-pointing baseline for the pointing game): {df.mask_frac.mean():.3f}",
         "", "| explanation | energy | n | pointing game | CAM energy in lesion | IoU (CAM >= 0.5 max) |", "|---|---|---|---|---|---|"]
    for e in ["LE", "DES", "all"]:
        d = df if e == "all" else df[df.energy == e]
        for name, pre in (("black box (true-class logit)", "bb"), ("top lesion concept", "concept")):
            L.append(f"| {name} | {e} | {len(d)} | {d[f'{pre}_point'].mean():.3f} | {d[f'{pre}_energy'].mean():.3f} | "
                     f"{d[f'{pre}_iou50'].mean():.3f} |")
    (run / "LOCALISATION.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
