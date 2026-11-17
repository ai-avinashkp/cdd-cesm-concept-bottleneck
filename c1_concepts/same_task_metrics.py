"""G2: our results re-expressed in the task definitions used by published CDD-CESM papers, for like-for-like
comparison. From the 3-class out-of-fold probabilities (breast level, patient-grouped folds):
  normal vs tumorous   P(tumorous) = P(benign) + P(malignant)            (accuracy at 0.5, AUC)
  normal vs malignant  on normal + malignant breasts, P(M) / (P(N) + P(M)) (AUC)
  benign vs malignant  on benign + malignant breasts, P(M) / (P(B) + P(M)) (AUC)
Mammo-CLIP seeds 0-2, black box and concept bottleneck; mean +- SD over seeds. Writes runs/SAME_TASK.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from compare_runs import CLASSES, TAB, model_probs

HERE = Path(__file__).resolve().parent
RUNS = ["b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2"]


def tasks(p, y):
    tum = p[:, 1] + p[:, 2]
    nm = (y == 0) | (y == 2)
    bm = y > 0
    return [accuracy_score(y > 0, tum >= 0.5), roc_auc_score(y > 0, tum),
            roc_auc_score(y[nm] == 2, p[nm, 2] / (p[nm, 0] + p[nm, 2])),
            roc_auc_score(y[bm] == 2, p[bm, 2] / (p[bm, 1] + p[bm, 2]))]


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    y = br.y.to_numpy()
    res = {"black box": [], "concept bottleneck": []}
    for r in RUNS:
        probs = model_probs(r, br)
        for kind in res:
            res[kind].append(tasks(probs[f"{r} / {kind}"], y))
    names = ["normal vs tumorous accuracy", "normal vs tumorous AUC", "normal vs malignant AUC", "benign vs malignant AUC"]
    L = ["# Our results in published task definitions (Mammo-CLIP, 3 seeds, breast level, patient-grouped CV)", "",
         "| model | " + " | ".join(names) + " |", "|---|---|---|---|---|"]
    for kind, rows in res.items():
        a = np.array(rows)
        L.append(f"| {kind} | " + " | ".join(f"{a[:, j].mean():.3f} ± {a[:, j].std(ddof=1):.3f}" for j in range(4)) + " |")
    text = "\n".join(L)
    (HERE / "runs" / "SAME_TASK.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
