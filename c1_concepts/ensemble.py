"""Seed ensemble of the three Mammo-CLIP runs (seeds 0-2): no extra training.
  black box: mean of the three label-head probabilities per breast
  CBM: mean of the three runs' breast-level concept probabilities -> logistic regression (other folds -> fold k)
Compared with each single seed (mean) and paired vs seed 0. Writes runs/ENSEMBLE.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from compare_runs import CLASSES, TAB, breast_level, metrics, model_probs

HERE = Path(__file__).resolve().parent
RUNS = ["b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2"]


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    y, folds = br.y.to_numpy(), br.fold.to_numpy()
    single = {r: model_probs(r, br) for r in RUNS}
    bb = np.mean([single[r][f"{r} / black box"] for r in RUNS], axis=0)
    conc = []
    for r in RUNS:
        pred, cols = breast_level(r)
        conc.append(pred.loc[br.index, cols].to_numpy())
    X = np.mean(conc, axis=0)
    cbm = np.zeros_like(bb)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        cbm[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])
    rows = {"ensemble (3 seeds) / black box": bb, "ensemble (3 seeds) / concept bottleneck": cbm}
    L = ["# Seed ensemble of Mammo-CLIP (3 runs), breast level, patient-grouped CV", "",
         "| model | accuracy | macro AUC | B-vs-M AUC |", "|---|---|---|---|"]
    for kind in ("black box", "concept bottleneck"):
        m = np.array([metrics(single[r][f"{r} / {kind}"], y) for r in RUNS])
        L.append(f"| single seed mean / {kind} | " + " | ".join(f"{v:.3f}" for v in m.mean(0)) + " |")
    for n, p in rows.items():
        L.append(f"| **{n}** | " + " | ".join(f"{v:.3f}" for v in metrics(p, y)) + " |")
    pats = br.patient.to_numpy()
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(2000)]
    L += ["", "## Paired differences, patient bootstrap x2000", "", "| comparison | accuracy | macro AUC | B-vs-M AUC |",
          "|---|---|---|---|"]
    pairs = [("ensemble CBM vs seed-0 CBM", cbm, single[RUNS[0]][f"{RUNS[0]} / concept bottleneck"]),
             ("ensemble black box vs seed-0 black box", bb, single[RUNS[0]][f"{RUNS[0]} / black box"]),
             ("ensemble CBM vs ensemble black box", cbm, bb)]
    for n, a, b in pairs:
        d = np.array([metrics(a[i], y[i]) - metrics(b[i], y[i]) for i in boots])
        est = metrics(a, y) - metrics(b, y)
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        L.append(f"| {n} | " + " | ".join(f"{est[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}] p={(d[:, j] <= 0).mean():.3f}"
                                          for j in range(3)) + " |")
    text = "\n".join(L)
    (HERE / "runs" / "ENSEMBLE.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
