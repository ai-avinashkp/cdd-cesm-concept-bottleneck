"""G4 + cross-energy concept bottleneck (CE-CBM) pilot analysis (2026-11-02). Breast level, patient-grouped folds.

Q1 Can the LE image predict the CONTRAST (DES) findings?  DES-concept AUC from
    (a) LE image, cross-energy model (g4_LE_cross)   (b) DES image, reference (b2_mammoclip_640x1024)
    (c) morphology baseline: logistic regression on the breast's LE *report* concepts (out-of-fold).
Q2 Iodine-trace test (risk B2): among breasts whose LE report states a mass, predict enhancement
    (des_mass_enh or des_nme) vs no enhancement. If the LE image beats the morphology baseline by a wide margin,
    contrast information may be leaking into LE (LE is acquired after injection).
Q3 LE-only diagnosis: does learning DES concepts help?  black box and CBM of g4_LE_cross vs g4_LE_only, plus the
    full LE+DES model as reference; paired patient bootstrap for CE-CBM vs LE-CBM.
Writes runs/G4_CROSS_ENERGY.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from compare_runs import CLASSES, TAB, metrics

HERE = Path(__file__).resolve().parent
CROSS, LEONLY, FULL = "g4_LE_cross", "g4_LE_only", "b2_mammoclip_640x1024"


def complete(run):
    return len(list((HERE / "runs" / run).glob("oof_fold*.csv"))) == 5


def breast_pred(run, energy):
    oof = pd.concat([pd.read_csv(f) for f in sorted((HERE / "runs" / run).glob("oof_fold*.csv"))])
    img = pd.read_csv(TAB / "images.csv")[["Image_name", "breast", "energy"]]
    d = oof.merge(img, on="Image_name")
    d = d[d.energy == energy] if energy else d
    pc = [c for c in d.columns if c.startswith("p_")]
    agg = {c: ("max" if c[2:] not in CLASSES else "mean") for c in pc}
    return d.groupby("breast").agg(agg)


def oof_lr(X, y, folds):
    p = np.zeros(len(y))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        p[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def cbm_probs(P, cols, y, folds):
    X = P[[f"p_{c}" for c in cols]].to_numpy()
    out = np.zeros((len(y), 3))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        out[te] = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])
    return out


def main():
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    true = pd.read_csv(HERE / "concepts_breast.csv").set_index("breast").loc[br.index]
    mask = pd.read_csv(HERE / "concept_mask_breast.csv").set_index("breast").loc[br.index]
    le = [c for c in true.columns if c.startswith("le_")]
    des = [c for c in true.columns if c.startswith("des_")]
    folds, y = br.fold.to_numpy(), br.y.to_numpy()
    L = ["# G4 / cross-energy concept bottleneck pilot (Mammo-CLIP B2, LE-only models)", ""]
    import os
    if not complete(CROSS) and not os.environ.get("G4_PARTIAL"):
        print("cross run not complete (set G4_PARTIAL=1 for a preliminary analysis on finished folds)"); return
    A = breast_pred(CROSS, "LE")
    if len(A) < len(br):                                   # preliminary: restrict to breasts in finished folds
        br = br.loc[br.index.intersection(A.index)]
        true, mask = true.loc[br.index], mask.loc[br.index]
        folds, y = br.fold.to_numpy(), br.y.to_numpy()
        L[0] += f" -- PRELIMINARY, {br.fold.nunique()}/5 folds, {len(br)} breasts"
    A = A.loc[br.index]
    R = breast_pred(FULL, "DES").reindex(br.index)

    L += ["## Q1. Predicting the contrast (DES) findings", "",
          "| DES concept | positives | morphology baseline (LE report concepts) | **LE image (cross-energy)** | DES image (reference) |",
          "|---|---|---|---|---|"]
    Xm = true[le].to_numpy(float)
    rows = []
    for c in des:
        ok = (mask[c] == 1).to_numpy()
        yc = true[c].to_numpy()[ok]
        if yc.sum() < 8:
            continue
        pm = oof_lr(Xm[ok], yc, folds[ok])
        a_m, a_l = roc_auc_score(yc, pm), roc_auc_score(yc, A[f"p_{c}"].to_numpy()[ok])
        a_d = roc_auc_score(yc, R[f"p_{c}"].fillna(0).to_numpy()[ok])
        rows.append((a_m, a_l, a_d))
        L.append(f"| {c} | {int(yc.sum())} | {a_m:.3f} | {a_l:.3f} | {a_d:.3f} |")
    r = np.array(rows)
    L.append(f"| **mean** | | {r[:, 0].mean():.3f} | {r[:, 1].mean():.3f} | {r[:, 2].mean():.3f} |")

    L += ["", "## Q2. Iodine-trace test: enhancement among LE-mass breasts", ""]
    sel = (true.le_mass == 1).to_numpy() & (mask.des_mass_enh == 1).to_numpy() & (mask.des_nme == 1).to_numpy()
    enh = ((true.des_mass_enh == 1) | (true.des_nme == 1)).to_numpy()[sel].astype(int)
    pm = oof_lr(Xm[sel], enh, folds[sel])
    score_l = np.maximum(A.p_des_mass_enh, A.p_des_nme).to_numpy()[sel]
    score_d = np.maximum(R.p_des_mass_enh, R.p_des_nme).fillna(0).to_numpy()[sel]
    L += [f"- breasts with an LE mass: {int(sel.sum())}; enhancing {int(enh.sum())}, non-enhancing {int((1 - enh).sum())}",
          f"- morphology baseline (LE report concepts): AUC {roc_auc_score(enh, pm):.3f}",
          f"- **LE image (cross-energy model): AUC {roc_auc_score(enh, score_l):.3f}**",
          f"- DES image (reference): AUC {roc_auc_score(enh, score_d):.3f}",
          "- Reading: if the LE image is far above the morphology baseline and close to the DES reference, contrast "
          "information is probably leaking into LE (risk B2); if it is close to the morphology baseline, the LE "
          "model is using morphology, as intended."]

    L += ["", "## Q3. LE-only diagnosis: does learning the contrast concepts help?", "",
          "| model | accuracy | macro AUC | B-vs-M AUC |", "|---|---|---|---|"]
    res = {}
    res["LE black box (cross-energy model)"] = A[[f"p_{c}" for c in CLASSES]].to_numpy()
    res["CE-CBM: LE + predicted DES concepts (cross)"] = cbm_probs(A, le + des, y, folds)
    res["CBM: LE concepts only (cross model)"] = cbm_probs(A, le, y, folds)
    if complete(LEONLY):
        B = breast_pred(LEONLY, "LE").loc[br.index]
        res["LE black box (standard LE model)"] = B[[f"p_{c}" for c in CLASSES]].to_numpy()
        res["CBM: LE concepts only (standard LE model)"] = cbm_probs(B, le, y, folds)
    F = breast_pred(FULL, None).loc[br.index]
    res["reference: LE+DES black box (full model)"] = F[[f"p_{c}" for c in CLASSES]].to_numpy()
    for n, p in res.items():
        p = p / p.sum(1, keepdims=True)
        L.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in metrics(p, y)) + " |")
    if complete(LEONLY):
        pats = br.patient.to_numpy()
        up = np.unique(pats)
        idx = {p: np.where(pats == p)[0] for p in up}
        rng = np.random.default_rng(0)
        a, b = res["CE-CBM: LE + predicted DES concepts (cross)"], res["CBM: LE concepts only (standard LE model)"]
        diffs = []
        for _ in range(2000):
            i = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
            diffs.append(metrics(a[i], y[i]) - metrics(b[i], y[i]))
        d = np.array(diffs)
        est = metrics(a, y) - metrics(b, y)
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        L += ["", "Paired patient bootstrap, CE-CBM minus LE-only CBM (standard model): " +
              "; ".join(f"{n} {e:+.3f} [{l:+.3f}, {h:+.3f}], P(<=0)={(d[:, j] <= 0).mean():.3f}"
                        for j, (n, e, l, h) in enumerate(zip(["accuracy", "macro AUC", "B-vs-M AUC"], est, lo, hi)))]
    text = "\n".join(L)
    (HERE / "runs" / "G4_CROSS_ENERGY.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
