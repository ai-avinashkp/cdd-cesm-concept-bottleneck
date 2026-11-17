"""Normal vs benign (and the other pairs) for the final 3-seed ensemble, breast level, patient-grouped CV.
Pairwise AUC uses P(b)/(P(a)+P(b)) on breasts of the two classes; 95% CI from 2,000 patient bootstrap resamples.
Writes runs/PAIRWISE.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from compare_runs import CLASSES, TAB, breast_level, model_probs

HERE = Path(__file__).resolve().parent
RUNS = ["b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2"]
PAIRS = [(0, 1, "normal vs benign"), (0, 2, "normal vs malignant"), (1, 2, "benign vs malignant")]


def main():
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
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(2000)]

    def pair_stats(p, a, b, ii):
        m = np.isin(y[ii], [a, b])
        yy, pp = y[ii][m], p[ii][m]
        s = pp[:, b] / (pp[:, a] + pp[:, b])
        pred_b = s >= 0.5
        auc = roc_auc_score(yy == b, s)
        acc = accuracy_score(yy == b, pred_b)
        sens_b = pred_b[yy == b].mean()          # recall of the second class
        spec_a = (~pred_b)[yy == a].mean()        # recall of the first class
        return np.array([auc, acc, spec_a, sens_b])

    L = ["# Pairwise metrics, 3-seed ensemble (breast level, patient-grouped CV; restricted to the two classes)", "",
         "Columns: AUC; accuracy at 0.5 of the two-class probability; recall of the first class; recall of the second "
         "class. 95% CI from 2,000 patient bootstrap resamples.", "",
         "| pair | model | n | AUC | accuracy | recall (first) | recall (second) |", "|---|---|---|---|---|---|---|"]
    allidx = np.arange(len(y))
    for a, b, name in PAIRS:
        n = int(np.isin(y, [a, b]).sum())
        for mname, p in (("concept bottleneck", cbm), ("black box", bb)):
            pt = pair_stats(p, a, b, allidx)
            bs = np.array([pair_stats(p, a, b, i) for i in boots])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            L.append(f"| {name} | {mname} | {n} | " + " | ".join(f"{pt[j]:.3f} ({lo[j]:.3f}-{hi[j]:.3f})" for j in range(4)) + " |")
        d = np.array([pair_stats(cbm, a, b, i)[0] - pair_stats(bb, a, b, i)[0] for i in boots])
        est = pair_stats(cbm, a, b, allidx)[0] - pair_stats(bb, a, b, allidx)[0]
        p2 = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        L.append(f"| {name} | AUC difference CBM - black box | | {est:+.3f} ({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}), p={p2:.3f} | | | |")
    text = "\n".join(L)
    (HERE / "runs" / "PAIRWISE.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
