"""C1: how much of the diagnosis do the report concepts explain? (the upper bound for a concept bottleneck)

Breast-level logistic regression on concept labels (concepts_breast.csv, extractor v2.1), using the shared
patient-grouped folds (pipeline/tables/folds.csv). Concept sets: LE only, DES only, both.
Targets: 3-class (Normal/Benign/Malignant) and benign-vs-malignant. Bootstrap 95% CIs over patients.
Writes concept_ceiling.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "pipeline" / "tables"
CLASSES = ["Normal", "Benign", "Malignant"]


def oof_predict(X, y, folds, C=1.0):
    proba = np.zeros((len(y), len(np.unique(y))))
    for k in sorted(np.unique(folds)):
        tr, te = folds != k, folds == k
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])
    return proba


def boot_ci(stat, patients, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    up = np.unique(patients)
    idx_by_p = {p: np.where(patients == p)[0] for p in up}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_by_p[p] for p in rng.choice(up, len(up))])
        try:
            vals.append(stat(idx))
        except ValueError:
            continue
    return np.percentile(vals, [2.5, 97.5])


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient")
    co = pd.read_csv(HERE / "concepts_breast.csv")
    df = br.merge(co, on="breast")
    le = [c for c in co.columns if c.startswith("le_")]
    des = [c for c in co.columns if c.startswith("des_")]
    sets = {"LE concepts only": le, "DES concepts only": des, "LE + DES concepts": le + des}
    y3 = df.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    folds, pats = df.fold.to_numpy(), df.patient.to_numpy()
    bm = df.label != "Normal"

    lines = ["# C1 concept ceiling (report concepts only, extractor v2.2)", "",
             f"- {len(df)} breasts, {df.patient.nunique()} patients; shared patient-grouped 5 folds; "
             "bootstrap 95% CI over patients (2,000 resamples)",
             "- This is an **upper bound**: concepts and labels come from the same radiologists (risk B4).", "",
             "| concept set | n concepts | 3-class accuracy | 3-class macro AUC | benign vs malignant AUC |",
             "|---|---|---|---|---|"]
    for name, cols in sets.items():
        X = df[cols].to_numpy(float)
        p3 = oof_predict(X, y3, folds)
        acc = accuracy_score(y3, p3.argmax(1))
        auc3 = roc_auc_score(y3, p3, multi_class="ovr", average="macro")
        Xb, yb = X[bm.to_numpy()], (df.label[bm] == "Malignant").to_numpy(int)
        pb = oof_predict(Xb, yb, folds[bm.to_numpy()])[:, 1]
        aucb = roc_auc_score(yb, pb)
        ci_acc = boot_ci(lambda i: accuracy_score(y3[i], p3[i].argmax(1)), pats)
        ci_auc3 = boot_ci(lambda i: roc_auc_score(y3[i], p3[i], multi_class="ovr", average="macro"), pats)
        pb_p = pats[bm.to_numpy()]
        ci_aucb = boot_ci(lambda i: roc_auc_score(yb[i], pb[i]), pb_p)
        lines.append(f"| {name} | {len(cols)} | {acc:.3f} [{ci_acc[0]:.3f}, {ci_acc[1]:.3f}] | "
                     f"{auc3:.3f} [{ci_auc3[0]:.3f}, {ci_auc3[1]:.3f}] | {aucb:.3f} [{ci_aucb[0]:.3f}, {ci_aucb[1]:.3f}] |")

    # interpretable weights: benign-vs-malignant model on all concepts, fitted on all abnormal breasts
    cols = le + des
    m = LogisticRegression(C=1.0, max_iter=5000).fit(df.loc[bm, cols].to_numpy(float),
                                                     (df.label[bm] == "Malignant").to_numpy(int))
    w = pd.Series(m.coef_[0], index=cols).sort_values()
    prev = df.loc[bm, cols].sum()
    lines += ["", "## Concept weights (benign vs malignant, all abnormal breasts; log-odds per concept)", "",
              "| concept | weight | breasts with concept (abnormal) |", "|---|---|---|"]
    for c in list(w.index[:8]) + list(w.index[-10:]):
        lines.append(f"| {c} | {w[c]:+.2f} | {int(prev[c])} |")
    lines += ["", "## Concept prevalence (all 566 breasts)", "", "| concept | breasts | Normal | Benign | Malignant |",
              "|---|---|---|---|---|"]
    for c in cols:
        vc = df.loc[df[c] == 1, "label"].value_counts()
        lines.append(f"| {c} | {int(df[c].sum())} | {vc.get('Normal', 0)} | {vc.get('Benign', 0)} | {vc.get('Malignant', 0)} |")
    text = "\n".join(lines)
    (HERE / "concept_ceiling.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
