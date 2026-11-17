"""Two-sided paired patient-bootstrap p-values (2 x min tail) for the manuscript's key comparisons.
Writes runs/TWO_SIDED.md. Same resampling as final_numbers.py (seed 0, 2,000 resamples).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from compare_runs import CLASSES, TAB, breast_level, metrics, model_probs

HERE = Path(__file__).resolve().parent
RUNS = ["b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2"]


def main(extra):
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    y, folds, pats = br.y.to_numpy(), br.fold.to_numpy(), br.patient.to_numpy()
    cols = breast_level(RUNS[0])[1]
    Xp = np.mean([breast_level(r)[0].loc[br.index, cols].to_numpy() for r in RUNS], axis=0)
    bb = np.mean([model_probs(r, br)[f"{r} / black box"] for r in RUNS], axis=0)
    cbm = np.zeros_like(bb)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        cbm[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Xp[tr], y[tr]).predict_proba(Xp[te])
    s0 = model_probs(RUNS[0], br)
    pairs = [("ensemble CBM vs ensemble black box", cbm, bb),
             ("ensemble CBM vs seed-0 CBM", cbm, s0[f"{RUNS[0]} / concept bottleneck"])]
    for r in extra:                                    # e.g. label-only black-box run
        lo = model_probs(r, br)[f"{r} / black box"]
        pairs += [(f"seed-0 CBM vs {r} black box (label-only)", s0[f"{RUNS[0]} / concept bottleneck"], lo),
                  (f"seed-0 multitask black box vs {r} black box", s0[f"{RUNS[0]} / black box"], lo)]
        pairs += [(f"{r} black box: point metrics", lo, None)]
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(2000)]
    L = ["# Two-sided paired bootstrap p-values (2,000 patient resamples)", "",
         "| comparison | accuracy | macro AUC | B-vs-M AUC |", "|---|---|---|---|"]
    for n, a, b in pairs:
        if b is None:
            L.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in metrics(a, y)) + " |")
            continue
        d = np.array([metrics(a[i], y[i]) - metrics(b[i], y[i]) for i in boots])
        est = metrics(a, y) - metrics(b, y)
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        p2 = np.minimum(1, 2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0)))
        L.append(f"| {n} | " + " | ".join(f"{est[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}] p2={p2[j]:.3f}" for j in range(3)) + " |")
    text = "\n".join(L)
    out = HERE / "runs" / ("TWO_SIDED.md" if not extra else "TWO_SIDED_with_labelonly.md")
    out.write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main(sys.argv[1:])
