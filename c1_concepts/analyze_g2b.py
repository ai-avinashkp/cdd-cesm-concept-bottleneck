"""G2b: dedicated binary normal-vs-tumorous model, LE only (the PMC12842785 setting), on our patient-grouped
folds. Breast-side level = mean P(tumorous) over the side's LE images. Uses whatever folds are finished.
Writes runs/<run>/G2B.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TAB = HERE.parent / "pipeline" / "tables"
RUN = sys.argv[1] if len(sys.argv) > 1 else "g2b_mammoclip_binary_LE"


def main():
    files = sorted((HERE / "runs" / RUN).glob("oof_fold*.csv"))
    oof = pd.concat([pd.read_csv(f).assign(fold=int(f.stem[-1])) for f in files])
    img = pd.read_csv(TAB / "images.csv")[["Image_name", "breast"]]
    br = pd.read_csv(TAB / "breasts.csv").set_index("breast")
    g = oof.merge(img, on="Image_name").groupby("breast").agg(p=("p_Tumorous", "mean"), fold=("fold", "first"))
    y = (br.loc[g.index, "label"] != "Normal").astype(int).to_numpy()
    p = g.p.to_numpy()
    L = [f"# G2b: binary normal vs tumorous, LE only, Mammo-CLIP, patient-grouped folds ({len(files)}/5 folds done)", "",
         f"- breast sides evaluated: {len(g)} (normal {int((y == 0).sum())}, tumorous {int((y == 1).sum())})",
         f"- **AUC {roc_auc_score(y, p):.3f}**; accuracy at 0.5: {accuracy_score(y, p >= 0.5):.3f}",
         "", "| fold | n | AUC | accuracy@0.5 |", "|---|---|---|---|"]
    for k, d in g.groupby("fold"):
        yk = (br.loc[d.index, "label"] != "Normal").astype(int).to_numpy()
        L.append(f"| {k} | {len(d)} | {roc_auc_score(yk, d.p):.3f} | {accuracy_score(yk, d.p >= 0.5):.3f} |")
    L += ["", "Reference: PMC12842785 (same 566-side cohort, LE only, dual-view transformer): AUC 0.992, accuracy "
          "0.969. Our 3-class model (LE+DES) re-expressed as normal vs tumorous: AUC 0.901 (CBM), 0.889 (black box)."]
    text = "\n".join(L)
    (HERE / "runs" / RUN / "G2B.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
