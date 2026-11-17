"""C1 section analyses on the 3-seed Mammo-CLIP ensemble (CPU only, 2026-11-02). Breast level, patient folds.

A. Clinical operating points for malignancy (malignant vs normal+benign; score = P(malignant)):
   sensitivity at specificity 0.80 / 0.90 and specificity at sensitivity 0.90 / 0.95, for the ensemble CBM and
   black box, with patient-bootstrap 95% CIs. The threshold is chosen on the pooled out-of-fold scores (reported
   as an operating-point description, not a deployment threshold).
B. G6 intervention: a radiologist corrects k concepts per breast (oracle = report value; excluded cells are
   never corrected). Policies:
     random          mean over 20 random orders
     uncertainty     concepts closest to 0.5 first (standard UCP baseline)
     importance      static order by the fold's head weights (sum |coef| x concept SD)
     expected impact per breast: E[change in the class-probability vector] = p_c*|f(c=1)-f| + (1-p_c)*|f(c=0)-f|
   Heads:
     sequential   logistic regression on PREDICTED concepts (as in the main results)
     independent  logistic regression on TRUE (report) concepts, applied to predicted concepts
Writes runs/G6_CLINICAL.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from compare_runs import CLASSES, TAB, breast_level, model_probs

HERE = Path(__file__).resolve().parent
RUNS = ["b2_mammoclip_640x1024", "b2_mammoclip_640x1024_s1", "b2_mammoclip_640x1024_s2"]
KS = [0, 1, 2, 3, 5, 8, 12, 35]


def op_points(y_mal, s):
    fpr, tpr, _ = roc_curve(y_mal, s)
    spec = 1 - fpr
    out = []
    for sp in (0.80, 0.90):
        out.append(tpr[spec >= sp].max() if (spec >= sp).any() else np.nan)
    for se in (0.90, 0.95):
        out.append(spec[tpr >= se].max() if (tpr >= se).any() else np.nan)
    return np.array(out)


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    y, folds, pats = br.y.to_numpy(), br.fold.to_numpy(), br.patient.to_numpy()
    true = pd.read_csv(HERE / "concepts_breast.csv").set_index("breast").loc[br.index]
    mask = pd.read_csv(HERE / "concept_mask_breast.csv").set_index("breast").loc[br.index]
    cols = list(true.columns)
    Xp = np.mean([breast_level(r)[0].loc[br.index, cols].to_numpy() for r in RUNS], axis=0)
    Xt, M = true[cols].to_numpy(float), mask[cols].to_numpy() == 1
    bb = np.mean([model_probs(r, br)[f"{r} / black box"] for r in RUNS], axis=0)

    # ---------- A. clinical operating points
    seq_heads, ind_heads = {}, {}
    cbm = np.zeros_like(bb)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        seq_heads[k] = LogisticRegression(C=1.0, max_iter=5000).fit(Xp[tr], y[tr])
        ind_heads[k] = LogisticRegression(C=1.0, max_iter=5000).fit(Xt[tr], y[tr])
        cbm[te] = seq_heads[k].predict_proba(Xp[te])
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(2000)]
    ym = (y == 2).astype(int)
    L = ["# C1 sections: clinical operating points (A) and G6 interventions (B); 3-seed Mammo-CLIP ensemble", "",
         "## A. Malignancy detection (malignant vs normal + benign), breast level", "",
         "| model | AUC | sens @ spec 0.80 | sens @ spec 0.90 | spec @ sens 0.90 | spec @ sens 0.95 |",
         "|---|---|---|---|---|---|"]
    for name, p in (("concept bottleneck", cbm), ("black box", bb)):
        s = p[:, 2]
        pt = op_points(ym, s)
        bs = np.array([op_points(ym[i], s[i]) for i in boots])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        auc = roc_auc_score(ym, s)
        L.append(f"| {name} | {auc:.3f} | " + " | ".join(f"{pt[j]:.3f} [{lo[j]:.2f}, {hi[j]:.2f}]" for j in range(4)) + " |")

    # ---------- B. interventions
    def apply(head, X):
        return head.predict_proba(X)

    def run_policy(policy, head_kind, k_list, seed=0):
        res = {k: np.zeros_like(bb) for k in k_list}
        rs = np.random.default_rng(seed)
        for f in np.unique(folds):
            te = folds == f
            head = (seq_heads if head_kind == "sequential" else ind_heads)[f]
            Xf, Tf, Mf = Xp[te], Xt[te], M[te]
            if policy == "uncertainty":
                score = -np.abs(Xf - 0.5)
            elif policy == "random":
                score = rs.random(Xf.shape)
            elif policy == "importance":
                imp = np.abs(head.coef_).sum(0) * Xp[folds != f].std(0)
                score = np.tile(imp, (len(Xf), 1))
            else:                                               # expected impact
                base = apply(head, Xf)
                score = np.zeros_like(Xf)
                for c in range(Xf.shape[1]):
                    X1, X0 = Xf.copy(), Xf.copy()
                    X1[:, c], X0[:, c] = 1, 0
                    d1 = np.abs(apply(head, X1) - base).sum(1)
                    d0 = np.abs(apply(head, X0) - base).sum(1)
                    score[:, c] = Xf[:, c] * d1 + (1 - Xf[:, c]) * d0
            score = np.where(Mf, score, -np.inf)                # never ask about excluded cells
            order = np.argsort(-score, axis=1)
            for k in k_list:
                Xi = Xf.copy()
                if k:
                    r = np.arange(len(Xi))[:, None]
                    sel = order[:, :k]
                    ok = Mf[r, sel]
                    Xi[r, sel] = np.where(ok, Tf[r, sel], Xi[r, sel])
                res[k][te] = apply(head, Xi)
        return res

    def summ(p):
        bm = y > 0
        return accuracy_score(y, p.argmax(1)), roc_auc_score(y[bm] == 2, p[bm, 2] / (p[bm, 1] + p[bm, 2]))

    L += ["", "## B. G6 interventions: accuracy (B-vs-M AUC) after correcting k concepts per breast", ""]
    for head_kind in ("sequential", "independent"):
        L += [f"### {head_kind} head", "", "| policy | " + " | ".join(f"k={k}" for k in KS) + " |",
              "|---|" + "---|" * len(KS)]
        for policy in ("random", "uncertainty", "importance", "expected impact"):
            if policy == "random":
                runs = [run_policy("random", head_kind, KS, seed=s) for s in range(20)]
                vals = {k: np.mean([summ(r[k]) for r in runs], axis=0) for k in KS}
            else:
                r = run_policy(policy, head_kind, KS)
                vals = {k: summ(r[k]) for k in KS}
            L.append(f"| {policy} | " + " | ".join(f"{vals[k][0]:.3f} ({vals[k][1]:.3f})" for k in KS) + " |")
        L.append("")
    L += ["Notes: k=35 = all usable concepts corrected. The independent head is trained on report concepts, so with "
          "full correction it reaches the concept ceiling; without correction it may be below the sequential head "
          "(train/test input mismatch)."]
    text = "\n".join(L)
    (HERE / "runs" / "G6_CLINICAL.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
