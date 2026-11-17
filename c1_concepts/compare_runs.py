"""Compare finished image runs at the breast level with patient-bootstrap 95% CIs and PAIRED differences.

For each run: black box (mean of the label head over the breast's images) and the sequential concept bottleneck
(logistic regression on predicted concepts, fitted on the other folds). Metrics: 3-class accuracy, 3-class macro
AUC, benign-vs-malignant AUC. The bootstrap resamples patients (2,000 times); paired differences use the same
resample for both models, so they account for the shared test set.

Usage: python compare_runs.py b1 b2_mammoclip_640x1024 [more runs]   -> runs/COMPARISON.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "pipeline" / "tables"
CLASSES = ["Normal", "Benign", "Malignant"]
B = 2000


def breast_level(run):
    oof = pd.concat([pd.read_csv(f) for f in sorted((HERE / "runs" / run).glob("oof_fold*.csv"))])
    img = pd.read_csv(TAB / "images.csv")[["Image_name", "breast", "energy"]]
    true = pd.read_csv(HERE / "concepts_breast.csv").set_index("breast")
    cols = [c for c in true.columns if f"p_{c}" in oof.columns]
    d = oof.merge(img, on="Image_name")
    rows = []
    for b, g in d.groupby("breast"):
        r = {"breast": b}
        for c in cols:
            v = g.loc[g.energy == ("LE" if c.startswith("le_") else "DES"), f"p_{c}"]
            r[c] = v.max() if len(v) else 0.0
        for c in CLASSES:
            r[f"bb_{c}"] = g[f"p_{c}"].mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("breast"), cols


def model_probs(run, br):
    pred, cols = breast_level(run)
    pred = pred.loc[br.index]
    y, folds = br.y.to_numpy(), br.fold.to_numpy()
    bb = pred[[f"bb_{c}" for c in CLASSES]].to_numpy()
    bb = bb / bb.sum(1, keepdims=True)
    X = pred[cols].to_numpy()
    cbm = np.zeros_like(bb)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        cbm[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])
    return {f"{run} / black box": bb, f"{run} / concept bottleneck": cbm}


def metrics(p, y):
    bm = y > 0
    return np.array([accuracy_score(y, p.argmax(1)), roc_auc_score(y, p, multi_class="ovr"),
                     roc_auc_score(y[bm] == 2, p[bm, 2] / (p[bm, 1] + p[bm, 2]))])


def main(runs):
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    models = {}
    for r in runs:
        models.update(model_probs(r, br))
    y, pats = br.y.to_numpy(), br.patient.to_numpy()
    up = np.unique(pats)
    idx_by_p = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx_by_p[p] for p in rng.choice(up, len(up))]) for _ in range(B)]
    point = {m: metrics(p, y) for m, p in models.items()}
    dist = {m: np.array([metrics(p[i], y[i]) for i in boots]) for m, p in models.items()}

    names = ["accuracy", "macro AUC", "B-vs-M AUC"]
    L = [f"# Model comparison (breast level, {len(y)} breasts, patient bootstrap x{B})", "",
         "| model | " + " | ".join(names) + " |", "|---|---|---|---|"]
    for m in models:
        lo, hi = np.percentile(dist[m], [2.5, 97.5], axis=0)
        L.append(f"| {m} | " + " | ".join(f"{point[m][j]:.3f} [{lo[j]:.3f}, {hi[j]:.3f}]" for j in range(3)) + " |")
    L += ["", "## Paired differences (A - B): estimate [95% CI], P(diff <= 0) under the bootstrap", "",
          "| A vs B | " + " | ".join(names) + " |", "|---|---|---|---|"]
    keys = list(models)
    pairs = [(f"{r} / concept bottleneck", f"{r} / black box") for r in runs]
    pairs += [(f"{r} / {kind}", f"{runs[0]} / {kind}") for r in runs[1:] for kind in ("black box", "concept bottleneck")]
    for a, b in pairs:
        if a not in keys or b not in keys:
            continue
        diff = dist[a] - dist[b]
        est = point[a] - point[b]
        lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
        pv = (diff <= 0).mean(axis=0)
        L.append(f"| {a} vs {b} | " + " | ".join(f"{est[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}], p={pv[j]:.3f}"
                                                for j in range(3)) + " |")
    L += ["", "Single training seed per run: the CIs cover test-set (patient) sampling, not training randomness."]
    text = "\n".join(L)
    (HERE / "runs" / "COMPARISON.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:] or ["b1", "b2_mammoclip_640x1024"])
