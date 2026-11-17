"""Step-1 analyses (2026-11-01).

A. Seeds: Mammo-CLIP at seeds 0, 1, 2 (patient-grouped folds). Breast-level black box and sequential concept
   bottleneck metrics per seed, then mean +- SD across seeds (training randomness), and the CBM - black box
   difference per seed.
B. Leakage: the same model and seed trained with image-level random folds (one patient's images can sit in
   train and test) vs the correct patient-grouped folds. Image-level metrics, the way most published CDD-CESM
   numbers are reported, plus breast-level.

Writes runs/STEP1_SEEDS_LEAKAGE.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from compare_runs import CLASSES, TAB, metrics, model_probs

HERE = Path(__file__).resolve().parent
SEEDS = {0: "b2_mammoclip_640x1024", 1: "b2_mammoclip_640x1024_s1", 2: "b2_mammoclip_640x1024_s2"}
LEAK = "b2_mammoclip_640x1024_imagesplit"
NAMES = ["accuracy", "macro AUC", "B-vs-M AUC"]


def complete(run):
    return len(list((HERE / "runs" / run).glob("oof_fold*.csv"))) == 5


def image_metrics(run):
    oof = pd.concat([pd.read_csv(f) for f in sorted((HERE / "runs" / run).glob("oof_fold*.csv"))])
    img = pd.read_csv(TAB / "images.csv")[["Image_name", "label"]]
    d = oof.merge(img, on="Image_name")
    y = d.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p = d[[f"p_{c}" for c in CLASSES]].to_numpy()
    bm = y > 0
    return np.array([accuracy_score(y, p.argmax(1)), roc_auc_score(y, p, multi_class="ovr"),
                     roc_auc_score(y[bm] == 2, p[bm, 2] / (p[bm, 1] + p[bm, 2]))])


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    y = br.y.to_numpy()
    L = ["# Step 1: seeds and leakage (Mammo-CLIP B2, 640x1024)", ""]

    done = {s: r for s, r in SEEDS.items() if complete(r)}
    L += [f"## A. Training-seed variability ({len(done)} seeds complete: {sorted(done)})", "",
          "| seed | model | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|"]
    per = {"black box": [], "concept bottleneck": []}
    for s, r in done.items():
        probs = model_probs(r, br)
        for kind in per:
            m = metrics(probs[f"{r} / {kind}"], y)
            per[kind].append(m)
            L.append(f"| {s} | {kind} | " + " | ".join(f"{v:.3f}" for v in m) + " |")
    if len(done) >= 2:
        L += ["", "| model | " + " | ".join(f"{n} mean ± SD" for n in NAMES) + " |", "|---|---|---|---|"]
        for kind, rows in per.items():
            a = np.array(rows)
            L.append(f"| {kind} | " + " | ".join(f"{a[:, j].mean():.3f} ± {a[:, j].std(ddof=1):.3f}"
                                                for j in range(3)) + " |")
        d = np.array(per["concept bottleneck"]) - np.array(per["black box"])
        L.append(f"| CBM − black box | " + " | ".join(f"{d[:, j].mean():+.3f} ± {d[:, j].std(ddof=1):.3f} "
                                                      f"(positive in {int((d[:, j] > 0).sum())}/{len(d)} seeds)"
                                                      for j in range(3)) + " |")

    L += ["", "## B. Leakage: image-level random split vs patient-grouped split (seed 0)", ""]
    if complete(LEAK) and complete(SEEDS[0]):
        fl = pd.read_csv(HERE / "runs" / LEAK / "folds_image_split.csv")
        shared = np.mean([len(set(fl.patient[fl.fold == k]) & set(fl.patient[fl.fold != k])) /
                          fl.patient[fl.fold == k].nunique() for k in range(5)])
        ok, bad = image_metrics(SEEDS[0]), image_metrics(LEAK)
        L += [f"Under the image-level split, on average {shared:.0%} of each test fold's patients also have images "
              f"in its training folds.", "",
              "| split | level | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|",
              "| patient-grouped (correct) | image | " + " | ".join(f"{v:.3f}" for v in ok) + " |",
              "| image-level random (leaky) | image | " + " | ".join(f"{v:.3f}" for v in bad) + " |",
              "| **inflation** | image | " + " | ".join(f"{b - o:+.3f}" for o, b in zip(ok, bad)) + " |"]
    else:
        L.append("(not complete yet)")
    text = "\n".join(L)
    (HERE / "runs" / "STEP1_SEEDS_LEAKAGE.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
