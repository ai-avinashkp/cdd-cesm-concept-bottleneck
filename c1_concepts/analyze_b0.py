"""Analyse an image-baseline run (python analyze_b0.py <run>): out-of-fold predictions at the breast level.

1. Concept prediction quality: AUC per concept (breast level; image probabilities -> max over the breast's
   images of that energy), against the breast-level report concepts.
2. Black box: breast label from the image label head (mean of class probabilities over the breast's images).
3. Concept bottleneck (sequential): logistic regression on PREDICTED concepts -> breast label, fitted on the
   other folds, tested on the held-out fold.
4. Intervention: replace the k most uncertain predicted concepts per breast with the report value
   (a radiologist correcting the model), k = 0, 1, 3, 5, all; refit-free (same fitted head).
Writes runs/b0/RESULTS.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "pipeline" / "tables"
RUN = HERE / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "b1")
CLASSES = ["Normal", "Benign", "Malignant"]


def main():
    oof = pd.concat([pd.read_csv(f) for f in sorted(RUN.glob("oof_fold*.csv"))])
    img = pd.read_csv(TAB / "images.csv")[["Image_name", "breast", "energy"]]
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient")
    true = pd.read_csv(HERE / "concepts_breast.csv").set_index("breast")
    mask = pd.read_csv(HERE / "concept_mask_breast.csv").set_index("breast")
    cols = [c for c in true.columns if f"p_{c}" in oof.columns]
    d = oof.merge(img, on="Image_name")
    rows = []
    for b, g in d.groupby("breast"):
        r = {"breast": b}
        for c in cols:
            e = "LE" if c.startswith("le_") else "DES"
            v = g.loc[g.energy == e, f"p_{c}"]
            r[c] = v.max() if len(v) else 0.0
        for c in CLASSES:
            r[f"bb_{c}"] = g[f"p_{c}"].mean()
        rows.append(r)
    pred = pd.DataFrame(rows).set_index("breast")
    br = br[br.breast.isin(pred.index)].set_index("breast")
    pred, true, mask = pred.loc[br.index], true.loc[br.index], mask.loc[br.index]
    y = br.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    folds = br.fold.to_numpy()
    nfold = len(np.unique(folds))

    L = [f"# Baseline B0 results (EfficientNet-B0, 512x768, {nfold} folds, {len(br)} breasts)", ""]
    # 1. concepts
    L += ["## 1. Concept prediction (breast level)", "", "| concept | positives | AUC |", "|---|---|---|"]
    aucs = {}
    for c in cols:
        ok = mask[c] == 1                              # excluded cells are not scored
        if 5 <= true.loc[ok, c].sum() <= ok.sum() - 5:
            aucs[c] = roc_auc_score(true.loc[ok, c], pred.loc[ok, c])
            L.append(f"| {c} | {int(true[c].sum())} | {aucs[c]:.3f} |")
    L.append(f"\nMean concept AUC: {np.mean(list(aucs.values())):.3f} over {len(aucs)} concepts with >= 5 positives")

    # 2. black box
    pbb = pred[[f"bb_{c}" for c in CLASSES]].to_numpy()
    pbb = pbb / pbb.sum(1, keepdims=True)
    bm = y > 0
    res = {"black box (label head)": (accuracy_score(y, pbb.argmax(1)),
                                      roc_auc_score(y, pbb, multi_class="ovr"),
                                      roc_auc_score(y[bm] == 2, pbb[bm, 2] / (pbb[bm, 1] + pbb[bm, 2])))}

    # 3. concept bottleneck on predicted concepts, and 4. interventions
    Xp, Xt = pred[cols].to_numpy(), true[cols].to_numpy(float)
    Mk = mask[cols].to_numpy() == 1                    # interventions only correct usable (non-excluded) cells
    p_cbm = np.zeros((len(y), 3))
    p_int = {k: np.zeros((len(y), 3)) for k in (1, 3, 5, "all")}
    p_true = np.zeros((len(y), 3))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = LogisticRegression(C=1.0, max_iter=5000).fit(Xp[tr], y[tr])
        p_cbm[te] = m.predict_proba(Xp[te])
        unc = np.where(Mk[te], -np.abs(Xp[te] - 0.5), -np.inf)   # most uncertain usable concepts first
        order = np.argsort(-unc, axis=1)
        for kk in p_int:
            Xi = Xp[te].copy()
            if kk == "all":
                Xi = np.where(Mk[te], Xt[te], Xp[te])
            else:
                rows_ = np.arange(Xi.shape[0])[:, None]
                Xi[rows_, order[:, :kk]] = Xt[te][rows_, order[:, :kk]]
            p_int[kk][te] = m.predict_proba(Xi)
        mt = LogisticRegression(C=1.0, max_iter=5000).fit(Xt[tr], y[tr])
        p_true[te] = mt.predict_proba(Xt[te])

    def summ(p):
        return (accuracy_score(y, p.argmax(1)), roc_auc_score(y, p, multi_class="ovr"),
                roc_auc_score(y[bm] == 2, p[bm, 2] / (p[bm, 1] + p[bm, 2])))
    res["concept bottleneck (predicted concepts)"] = summ(p_cbm)
    for kk, p in p_int.items():
        res[f"  + radiologist corrects {kk} most uncertain concept(s)"] = summ(p)
    res["upper bound: head trained on report concepts"] = summ(p_true)
    L += ["", "## 2–4. Diagnosis (breast level, out-of-fold)", "",
          "| model | 3-class accuracy | 3-class macro AUC | benign vs malignant AUC |", "|---|---|---|---|"]
    for n, (a, au, b) in res.items():
        L.append(f"| {n} | {a:.3f} | {au:.3f} | {b:.3f} |")
    text = "\n".join(L)
    (RUN / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
