"""Like-for-like black box (audit 2026-11-16, finding F1). Writes runs/MATCHED_BLACK_BOX.md.

The averaged black box is the mean of image-label class probabilities and uses no breast labels, while the concept
head is a logistic regression fitted on breast labels. Here the black box gets the same second stage: a multinomial
logistic regression (L2, C=1), cross-fitted on the same folds, on breast-level aggregates of its class probabilities.
Main version: 12 inputs (mean and max of each class probability over the breast's LE images and over its recombined
images). Sensitivity: 6 inputs (means per image type) and 3 inputs (means over all images).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from compare_runs import B, CLASSES, TAB, breast_level

HERE = Path(__file__).resolve().parent
RUNS = ["b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2"]
LABEL_ONLY = "b6_mammoclip_labelonly"
PC = [f"p_{c}" for c in CLASSES]
NAMES = ["accuracy", "macro AUC", "B-vs-M AUC"]


def bb_features(run, breasts):
    """Per breast: [LE mean(3), LE max(3), DES mean(3), DES max(3)], overall mean(3)."""
    oof = pd.concat([pd.read_csv(f) for f in sorted((HERE / "runs" / run).glob("oof_fold*.csv"))])
    d = oof.merge(pd.read_csv(TAB / "images.csv")[["Image_name", "breast", "energy"]], on="Image_name")
    f12, f3 = [], []
    for b in breasts:
        g = d[d.breast == b]
        row = []
        for e in ("LE", "DES"):
            s = g[g.energy == e] if (g.energy == e).any() else g
            row += list(s[PC].mean()) + list(s[PC].max())
        f12.append(row)
        f3.append(list(g[PC].mean()))
    return np.array(f12), np.array(f3)


def crossfit(X, y, folds):
    out = np.zeros((len(y), 3))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        out[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])
    return out


def metrics(p, y):
    bm = y > 0
    return np.array([(p.argmax(1) == y).mean(), roc_auc_score(y, p, multi_class="ovr"),
                     roc_auc_score(y[bm] == 2, p[bm, 2] / (p[bm, 1] + p[bm, 2]))])


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    y = br.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    folds, pats = br.fold.to_numpy(), br.patient.to_numpy()
    img = pd.read_csv(TAB / "images.csv")
    discordant = (img.groupby("breast").label.nunique() > 1).reindex(br.index).to_numpy()

    def outputs(runs):
        cols = breast_level(runs[0])[1]
        con = np.mean([breast_level(r)[0].loc[br.index, cols].to_numpy() for r in runs], axis=0)
        feats = [bb_features(r, br.index) for r in runs]
        f12, f3 = np.mean([f[0] for f in feats], axis=0), np.mean([f[1] for f in feats], axis=0)
        f6 = f12[:, [0, 1, 2, 6, 7, 8]]
        return {"concept bottleneck": crossfit(con, y, folds),
                "black box, averaged": f3 / f3.sum(1, keepdims=True),
                "black box, fitted (12 inputs)": crossfit(f12, y, folds),
                "black box, fitted (6 inputs)": crossfit(f6, y, folds),
                "black box, fitted (3 inputs)": crossfit(f3, y, folds)}

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

    M = outputs(RUNS)
    main3 = ["concept bottleneck", "black box, averaged", "black box, fitted (12 inputs)"]
    L = ["# Like-for-like black box (three-seed ensemble, 566 breasts, patient-grouped CV)", "",
         "The averaged black box is the mean of image-label class probabilities. The fitted black box adds a "
         "multinomial logistic regression (L2, C=1) on breast-level aggregates of those probabilities, cross-fitted on "
         "the same folds as the concept head. 12 inputs: mean and max of each class probability over the LE images and "
         "over the recombined images. 95% CI and two-sided p: 2,000 patient bootstrap resamples (paired).", "",
         "| output | " + " | ".join(NAMES) + " |", "|---|---|---|---|"]
    for m, p in M.items():
        pt, (lo, hi) = metrics(p, y), ci(p)
        L.append(f"| {m} | " + " | ".join(f"{pt[j]:.3f} ({lo[j]:.3f}-{hi[j]:.3f})" for j in range(3)) + " |")
    L += ["", "## Paired differences", "", "| A minus B | " + " | ".join(NAMES) + " |", "|---|---|---|---|"]
    for a, b in [("concept bottleneck", "black box, fitted (12 inputs)"), ("concept bottleneck", "black box, averaged"),
                 ("black box, fitted (12 inputs)", "black box, averaged"),
                 ("concept bottleneck", "black box, fitted (6 inputs)"),
                 ("concept bottleneck", "black box, fitted (3 inputs)")]:
        L.append(f"| {a} minus {b} | {paired(M[a], M[b])} |")

    L += ["", "## Errors by class (argmax)", "",
          "| output | normal recall | benign recall | malignant recall | benign called normal |", "|---|---|---|---|---|"]
    for m in main3:
        q = M[m].argmax(1)
        L.append(f"| {m} | " + " | ".join(f"{(q[y == c] == c).mean():.3f}" for c in range(3))
                 + f" | {int(((y == 1) & (q == 0)).sum())} |")

    L += ["", f"## Breasts whose image labels disagree ({int(discordant.sum())}) and the others "
              f"({int((~discordant).sum())}): accuracy", "", "| output | disagreeing | others |", "|---|---|---|"]
    for m in main3:
        ok = M[m].argmax(1) == y
        L.append(f"| {m} | {ok[discordant].mean():.3f} | {ok[~discordant].mean():.3f} |")

    L += ["", "## Malignancy detection (malignant vs normal or benign), score = P(malignant)", "",
          "| output | AUC | sens @ spec 0.80 | spec @ sens 0.90 |", "|---|---|---|---|"]
    for m in main3:
        fpr, tpr, _ = roc_curve(y == 2, M[m][:, 2])
        L.append(f"| {m} | {roc_auc_score(y == 2, M[m][:, 2]):.3f} | {tpr[1 - fpr >= 0.80].max():.3f} | "
                 f"{(1 - fpr)[tpr >= 0.90].max():.3f} |")

    ct = pd.crosstab(img.breast.map(br.label), img.label).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    L += ["", "## Image labels by breast label (images)", "",
          "| breast label | " + " | ".join(f"image {c}" for c in CLASSES) + " |", "|---|---|---|---|"]
    L += [f"| {c} | " + " | ".join(str(int(v)) for v in ct.loc[c]) + " |" for c in CLASSES]

    L += ["", "## Each seed separately", "", "| seed | output | " + " | ".join(NAMES) + " |", "|---|---|---|---|---|"]
    per, mach = [], []
    machine = br.machine.to_numpy()
    for s, r in enumerate(RUNS):
        Ms = outputs([r])
        per.append(metrics(Ms["concept bottleneck"], y) - metrics(Ms["black box, fitted (12 inputs)"], y))
        mach.append({(m, k): metrics(Ms[m][machine == k], y[machine == k]) for m in main3 for k in (1, 2)})
        for m in main3:
            L.append(f"| {s} | {m} | " + " | ".join(f"{v:.3f}" for v in metrics(Ms[m], y)) + " |")
        if s == 0:
            seed0 = Ms
    per = np.array(per)
    L += ["", "Concept bottleneck minus fitted black box, per seed: "
          + "; ".join(f"{n} " + ", ".join(f"{v:+.3f}" for v in per[:, j]) + f" (mean {per[:, j].mean():+.3f})"
                      for j, n in enumerate(NAMES)),
          "", "## By machine (means over the three seeds; accuracy / B-vs-M AUC)", "",
          "| output | machine 1 | machine 2 |", "|---|---|---|"]
    for m in main3:
        v = {k: np.mean([d[(m, k)] for d in mach], axis=0) for k in (1, 2)}
        L.append(f"| {m} | {v[1][0]:.3f} / {v[1][2]:.3f} | {v[2][0]:.3f} / {v[2][2]:.3f} |")

    lo_ = outputs([LABEL_ONLY])
    L += ["", "## Label-only network (no concept loss), seed 0", "",
          "| output | " + " | ".join(NAMES) + " |", "|---|---|---|---|"]
    for m in ("black box, averaged", "black box, fitted (12 inputs)"):
        L.append(f"| label-only {m} | " + " | ".join(f"{v:.3f}" for v in metrics(lo_[m], y)) + " |")
    L += ["", "| A minus B | " + " | ".join(NAMES) + " |", "|---|---|---|---|",
          f"| seed-0 concept bottleneck minus label-only fitted black box | "
          f"{paired(seed0['concept bottleneck'], lo_['black box, fitted (12 inputs)'])} |",
          f"| seed-0 fitted black box minus label-only fitted black box | "
          f"{paired(seed0['black box, fitted (12 inputs)'], lo_['black box, fitted (12 inputs)'])} |"]
    text = "\n".join(L)
    (HERE / "runs" / "MATCHED_BLACK_BOX.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
