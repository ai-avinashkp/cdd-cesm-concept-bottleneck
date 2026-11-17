"""Calibration of the three breast-level outputs (three-seed ensemble). Writes runs/CALIBRATION.md.

Brier score: mean over breasts of sum_c (p_c - 1[y = c])^2 (three classes; 0 is perfect, lower is better).
ECE: top-label expected calibration error, 10 equal-width confidence bins.
95% CI and two-sided paired p from 2,000 patient bootstrap resamples (same resamples as the other analyses).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from compare_runs import B, CLASSES, TAB, breast_level
from matched_black_box import RUNS, bb_features, crossfit

HERE = Path(__file__).resolve().parent


def brier(p, y):
    return ((p - np.eye(3)[y]) ** 2).sum(1).mean()


def ece(p, y, bins=10):
    conf, ok = p.max(1), p.argmax(1) == y
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    y = br.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    folds, pats = br.fold.to_numpy(), br.patient.to_numpy()
    cols = breast_level(RUNS[0])[1]
    con = np.mean([breast_level(r)[0].loc[br.index, cols].to_numpy() for r in RUNS], axis=0)
    feats = [bb_features(r, br.index) for r in RUNS]
    f12, f3 = np.mean([f[0] for f in feats], axis=0), np.mean([f[1] for f in feats], axis=0)
    M = {"concept bottleneck": crossfit(con, y, folds), "black box, fitted": crossfit(f12, y, folds),
         "black box, averaged": f3 / f3.sum(1, keepdims=True)}
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(B)]
    f = lambda p, yy: np.array([brier(p, yy), ece(p, yy)])  # noqa: E731
    L = ["# Calibration of the breast-level outputs (three-seed ensemble, 566 breasts)", "",
         "Brier score (three classes, lower is better) and top-label ECE (10 bins). 95% CI from 2,000 patient "
         "bootstrap resamples.", "", "| output | Brier | ECE |", "|---|---|---|"]
    for m, p in M.items():
        d = np.array([f(p[i], y[i]) for i in boots])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        pt = f(p, y)
        L.append(f"| {m} | {pt[0]:.3f} ({lo[0]:.3f}-{hi[0]:.3f}) | {pt[1]:.3f} ({lo[1]:.3f}-{hi[1]:.3f}) |")
    L += ["", "| A minus B | Brier | ECE |", "|---|---|---|"]
    for a, b in [("concept bottleneck", "black box, fitted"), ("concept bottleneck", "black box, averaged")]:
        d = np.array([f(M[a][i], y[i]) - f(M[b][i], y[i]) for i in boots])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        p2 = np.minimum(1, 2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0)))
        est = f(M[a], y) - f(M[b], y)
        L.append(f"| {a} minus {b} | " + " | ".join(
            f"{est[j]:+.3f} ({lo[j]:+.3f} to {hi[j]:+.3f}); two-sided p={p2[j]:.3f}" for j in range(2)) + " |")
    text = "\n".join(L)
    (HERE / "runs" / "CALIBRATION.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
