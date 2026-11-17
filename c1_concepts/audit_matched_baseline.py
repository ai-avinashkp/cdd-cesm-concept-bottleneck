"""Independent audit of the C1 headline numbers from the raw out-of-fold CSVs (does not import project code).

1. Integrity: every image predicted exactly once per run; each oof_fold file holds only patients of that fold.
2. Recompute per-seed and ensemble metrics for the paper's two outputs.
3. Fairness test: the paper's black box is the mean of IMAGE-label class probabilities, while the concept head is
   a logistic regression fitted on BREAST labels. Give the black box the same second stage (logistic regression on
   breast-level aggregates of its class probabilities, same cross-fitting) and compare again.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R4 = Path(sys.argv[1])
RUNS = [R4 / "c1_concepts" / "runs" / r for r in
        ("b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2")]
CL = ["Normal", "Benign", "Malignant"]

img = pd.read_csv(R4 / "pipeline" / "tables" / "images.csv")
br = pd.read_csv(R4 / "pipeline" / "tables" / "breasts.csv")
fo = pd.read_csv(R4 / "pipeline" / "tables" / "folds.csv")
br = br.merge(fo, on="patient").sort_values("breast").reset_index(drop=True)
y = br.label.map({c: i for i, c in enumerate(CL)}).to_numpy()
folds, pats = br.fold.to_numpy(), br.patient.to_numpy()
pfold = dict(zip(fo.patient, fo.fold))
print(f"breasts {len(br)}, patients {br.patient.nunique()}, images {len(img)}, labels {np.bincount(y)}")

# ---------- 1. integrity
for run in RUNS:
    seen = []
    for k in range(5):
        d = pd.read_csv(run / f"oof_fold{k}.csv").merge(img[["Image_name", "patient"]], on="Image_name")
        bad = (d.patient.map(pfold) != k).sum()
        assert bad == 0, (run.name, k, bad)
        seen += list(d.Image_name)
    assert len(seen) == len(set(seen)) == len(img), (run.name, len(seen), len(set(seen)))
print("integrity OK: each image predicted once per run, and only by the model that did not train on its patient")
n_pat_folds = fo.groupby("patient").fold.nunique().max()
print("max folds per patient:", n_pat_folds)


def load(run):
    d = pd.concat([pd.read_csv(run / f"oof_fold{k}.csv") for k in range(5)])
    return d.merge(img[["Image_name", "breast", "energy", "label"]], on="Image_name")


def aggregates(d):
    ccols = [c for c in d.columns if c.startswith("p_le_") or c.startswith("p_des_")]
    pc = [f"p_{c}" for c in CL]
    con, bbmean, bbfeat = [], [], []
    for b in br.breast:
        g = d[d.breast == b]
        le, de = g[g.energy == "LE"], g[g.energy == "DES"]
        con.append([(le if c.startswith("p_le_") else de)[c].max() if len(le if c.startswith("p_le_") else de) else 0.0
                    for c in ccols])
        m = g[pc].mean().to_numpy()
        bbmean.append(m / m.sum())
        f = []
        for e in (le, de):
            src = e if len(e) else g
            f += list(src[pc].mean()) + list(src[pc].max())
        bbfeat.append(f)
    return np.array(con), np.array(bbmean), np.array(bbfeat)


def crossfit(X):
    out = np.zeros((len(y), 3))
    for k in range(5):
        tr, te = folds != k, folds == k
        out[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])
    return out


def pair_auc(p, yy, a, b):
    m = np.isin(yy, [a, b])
    return roc_auc_score(yy[m] == b, p[m, b] / (p[m, a] + p[m, b]))


def metrics(p, yy):
    return np.array([(p.argmax(1) == yy).mean(), roc_auc_score(yy, p, multi_class="ovr"),
                     pair_auc(p, yy, 1, 2), pair_auc(p, yy, 0, 1), (p.argmax(1) == yy)[yy == 1].mean()])


NAMES = ["acc", "macroAUC", "BvM AUC", "NvB AUC", "benign recall"]
per = [aggregates(load(r)) for r in RUNS]

print("\n--- 2. per-seed (paper Table: CBM seq head vs black-box mean)")
for s, (con, bbm, bbf) in enumerate(per):
    print(f"seed {s}  black box {np.round(metrics(bbm, y), 3)}  CBM {np.round(metrics(crossfit(con), y), 3)}"
          f"  stacked black box {np.round(metrics(crossfit(bbf), y), 3)}")

con = np.mean([p[0] for p in per], 0)
bbm = np.mean([p[1] for p in per], 0)
bbf = np.mean([p[2] for p in per], 0)
M = {"CBM (paper)": crossfit(con), "black box mean (paper)": bbm, "black box stacked (same 2nd stage)": crossfit(bbf),
     "concepts + black-box features": crossfit(np.hstack([con, bbf]))}
# fit-free alternative: use the energy whose mean P(normal) is lowest (most abnormal view decides)
print("\n--- 3. ensemble")
print("columns:", NAMES)
for n, p in M.items():
    print(f"{n:38s} {np.round(metrics(p, y), 3)}")

up = np.unique(pats)
idx = {p: np.where(pats == p)[0] for p in up}
rng = np.random.default_rng(0)
boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(2000)]


def paired(a, b):
    d = np.array([metrics(M[a][i], y[i]) - metrics(M[b][i], y[i]) for i in boots])
    est = metrics(M[a], y) - metrics(M[b], y)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    p2 = np.minimum(1, 2 * np.minimum((d <= 0).mean(0), (d >= 0).mean(0)))
    print(f"\n{a}  minus  {b}")
    for j, n in enumerate(NAMES):
        print(f"  {n:14s} {est[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}]  two-sided p={p2[j]:.3f}")


paired("CBM (paper)", "black box mean (paper)")
paired("CBM (paper)", "black box stacked (same 2nd stage)")
paired("black box stacked (same 2nd stage)", "black box mean (paper)")

# ---------- breasts whose image labels disagree (benign on LE, normal on DES)
dis = ((br.label_le != br.label_des)).to_numpy()
print(f"\nbreasts with LE label != DES label: {dis.sum()};  breast label there: {dict(br.label[dis].value_counts())}")
for n, p in M.items():
    print(f"  {n:38s} accuracy on those {((p.argmax(1) == y)[dis]).mean():.3f}   on the other {(~dis).sum()}: "
          f"{((p.argmax(1) == y)[~dis]).mean():.3f}")
for n, p in M.items():
    cm = np.zeros((3, 3), int)
    for t, q in zip(y, p.argmax(1)):
        cm[t, q] += 1
    print(n, cm.tolist())
