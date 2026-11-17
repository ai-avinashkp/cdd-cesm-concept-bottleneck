"""Check (a) of 2026-11-17: the same network trained with breast labels (every image learns its breast's label,
--label-source breast; 93 of 2,006 image labels change). Three seeds, same folds, run on the NITJ H100 HPC.
Writes runs/BREAST_LABEL.md (all three seeds) or runs/BREAST_LABEL_seeds<k>.md for a subset:
  python breast_label_check.py        or   python breast_label_check.py 0

Question: once the class head no longer learns "normal" for images of benign breasts, does the black box
catch up with or overtake the concept model, and does the concept model change?
"""
from pathlib import Path

import numpy as np
import pandas as pd

from compare_runs import B, CLASSES, TAB, breast_level
from matched_black_box import RUNS, bb_features, crossfit, metrics

HERE = Path(__file__).resolve().parent
NEW = ["b2_breastlabel_s0", "b2_breastlabel_s1", "b2_breastlabel_s2"]
NAMES = ["accuracy", "macro AUC", "B-vs-M AUC"]


def outputs(runs, br, y, folds):
    cols = breast_level(runs[0])[1]
    con = np.mean([breast_level(r)[0].loc[br.index, cols].to_numpy() for r in runs], axis=0)
    feats = [bb_features(r, br.index) for r in runs]
    f12, f3 = np.mean([f[0] for f in feats], axis=0), np.mean([f[1] for f in feats], axis=0)
    return {"concept bottleneck": crossfit(con, y, folds), "black box, fitted": crossfit(f12, y, folds),
            "black box, averaged": f3 / f3.sum(1, keepdims=True)}


def main():
    import sys
    seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    global NEW
    new_runs, old_runs = [NEW[k] for k in seeds], [RUNS[k] for k in seeds]
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    y = br.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    folds, pats = br.fold.to_numpy(), br.patient.to_numpy()
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(B)]

    def ci(p):
        d = np.array([metrics(p[i], y[i]) for i in boots])
        return np.percentile(d, [2.5, 97.5], axis=0)

    def paired(pa, pb):
        d = np.array([metrics(pa[i], y[i]) - metrics(pb[i], y[i]) for i in boots])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        p2 = np.minimum(1.0, 2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0)))
        est = metrics(pa, y) - metrics(pb, y)
        return " | ".join(f"{est[j]:+.3f} ({lo[j]:+.3f} to {hi[j]:+.3f}); two-sided p={p2[j]:.3f}" for j in range(3))

    old, new = outputs(old_runs, br, y, folds), outputs(new_runs, br, y, folds)
    L = [f"# Breast-label training (check a; seeds {seeds}, ensemble over them; 566 breasts, same folds)", "",
         "Network as in the paper, but the class loss uses the breast label for every image. Run on one H100 "
         "(NITJ HPC); the image-label runs were trained on the laptop GPU. 95% CI and two-sided p: 2,000 patient "
         "bootstrap resamples (paired).", "", "| network / output | " + " | ".join(NAMES) + " |", "|---|---|---|---|"]
    for tag, M in (("image labels (paper)", old), ("breast labels", new)):
        for m, p in M.items():
            pt, (lo, hi) = metrics(p, y), ci(p)
            L.append(f"| {tag} / {m} | " + " | ".join(f"{pt[j]:.3f} ({lo[j]:.3f}-{hi[j]:.3f})" for j in range(3)) + " |")
    L += ["", "## Paired differences", "", "| A minus B | " + " | ".join(NAMES) + " |", "|---|---|---|---|"]
    pairs = [("breast labels / concept bottleneck", new["concept bottleneck"], "breast labels / black box, fitted", new["black box, fitted"]),
             ("breast labels / concept bottleneck", new["concept bottleneck"], "breast labels / black box, averaged", new["black box, averaged"]),
             ("image labels / concept bottleneck (paper)", old["concept bottleneck"], "breast labels / black box, averaged", new["black box, averaged"]),
             ("image labels / concept bottleneck (paper)", old["concept bottleneck"], "breast labels / black box, fitted", new["black box, fitted"]),
             ("breast labels / black box, averaged", new["black box, averaged"], "image labels / black box, averaged", old["black box, averaged"]),
             ("breast labels / concept bottleneck", new["concept bottleneck"], "image labels / concept bottleneck (paper)", old["concept bottleneck"])]
    for na, pa, nb, pb in pairs:
        L.append(f"| {na} minus {nb} | {paired(pa, pb)} |")
    L += ["", "## Benign recall and benign breasts called normal", "", "| network / output | benign recall | benign called normal |",
          "|---|---|---|"]
    for tag, M in (("image labels", old), ("breast labels", new)):
        for m, p in M.items():
            q = p.argmax(1)
            L.append(f"| {tag} / {m} | {(q[y == 1] == 1).mean():.3f} | {int(((y == 1) & (q == 0)).sum())} |")
    L += ["", "## Each seed of the breast-label network", "", "| seed | output | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|"]
    for s, r in zip(seeds, new_runs):
        Ms = outputs([r], br, y, folds)
        for m, p in Ms.items():
            L.append(f"| {s} | {m} | " + " | ".join(f"{v:.3f}" for v in metrics(p, y)) + " |")
    text = "\n".join(L)
    name = "BREAST_LABEL.md" if seeds == [0, 1, 2] else f"BREAST_LABEL_seeds{''.join(map(str, seeds))}.md"
    (HERE / "runs" / name).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
