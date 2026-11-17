"""Precision, recall, F1 and confusion matrices for the final 3-seed ensemble (CBM vs black box), breast level,
patient-grouped CV. Covers the 3-class task (argmax) and the three class pairs (P(b)/(P(a)+P(b)) >= 0.5).
95% CI from 2,000 patient bootstrap resamples. Writes runs/PRECISION_RECALL.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from compare_runs import CLASSES, TAB, breast_level, model_probs

HERE = Path(__file__).resolve().parent
RUNS = ["b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def setup():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    y, folds, pats = br.y.to_numpy(), br.fold.to_numpy(), br.patient.to_numpy()
    cols = breast_level(RUNS[0])[1]
    X = np.mean([breast_level(r)[0].loc[br.index, cols].to_numpy() for r in RUNS], axis=0)
    bb = np.mean([model_probs(r, br)[f"{r} / black box"] for r in RUNS], axis=0)
    cbm = np.zeros_like(bb)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        cbm[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])
    return y, pats, {"concept bottleneck": cbm, "black box": bb}


def predictions(p, y, pair):
    """Return (true labels, predicted labels, class ids) for the 3-class task or one pair."""
    if pair is None:
        return y, p.argmax(1), [0, 1, 2]
    a, b = pair
    m = np.isin(y, [a, b])
    s = p[m, b] / (p[m, a] + p[m, b])
    return y[m], np.where(s >= 0.5, b, a), [a, b]


def stats(yt, yp, labels):
    """accuracy, then precision/recall/F1 per class, then macro F1."""
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return np.concatenate([[(yt == yp).mean()], np.ravel(np.column_stack([pr, rc, f1])), [f1.mean()]])


def main():
    y, pats, models = setup()
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(2000)]

    L = ["# Precision, recall and F1, 3-seed ensemble (breast level, patient-grouped CV)", "",
         "Precision = of the breasts the model called X, the share that really are X. Recall = of the breasts that "
         "really are X, the share the model called X. F1 = harmonic mean of the two. 3-class: argmax; pairs: "
         "restricted to the two classes, cut-off 0.5 on P(b)/(P(a)+P(b)). 95% CI: 2,000 patient bootstrap resamples.", ""]
    for pair in [None] + PAIRS:
        labels = [0, 1, 2] if pair is None else list(pair)
        title = "3-class (normal / benign / malignant)" if pair is None else f"{CLASSES[pair[0]]} vs {CLASSES[pair[1]]}"
        n = len(y) if pair is None else int(np.isin(y, pair).sum())
        L += [f"## {title} (n = {n})", "",
              "| model | class | precision | recall | F1 |", "|---|---|---|---|---|"]
        cms = []
        for mname, p in models.items():
            yt, yp, _ = predictions(p, y, pair)
            pt = stats(yt, yp, labels)
            bs = np.array([stats(*predictions(p[i], y[i], pair)[:2], labels) for i in boots])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            f = lambda j: f"{pt[j]:.3f} ({lo[j]:.3f}-{hi[j]:.3f})"
            for c_i, c in enumerate(labels):
                j = 1 + 3 * c_i
                L.append(f"| {mname} | {CLASSES[c]} | {f(j)} | {f(j + 1)} | {f(j + 2)} |")
            L.append(f"| {mname} | **accuracy {f(0)}; macro F1 {f(len(pt) - 1)}** | | | |")
            cms.append((mname, confusion_matrix(yt, yp, labels=labels)))
        L.append("")
        for mname, cm in cms:
            L += [f"Confusion matrix, {mname} (rows = true, columns = predicted):", "",
                  "| true \\ predicted | " + " | ".join(CLASSES[c] for c in labels) + " |",
                  "|---" * (len(labels) + 1) + "|"]
            L += [f"| {CLASSES[c]} | " + " | ".join(str(v) for v in row) + " |" for c, row in zip(labels, cm)]
            L.append("")
    text = "\n".join(L)
    (HERE / "runs" / "PRECISION_RECALL.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
