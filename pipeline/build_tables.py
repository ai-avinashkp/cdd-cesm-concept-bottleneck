"""Shared data pipeline for r4 (C1 concept bottleneck, N1 contrast necessity). Fresh code, CDD-CESM only.

Builds three tables in ./tables/:
  images.csv   one row per image: path, patient, breast, energy (LE/DES), view, image-level label, BI-RADS
               (max over '$'-separated lesions), ACR density, machine, age, report text
  breasts.csv  one row per breast (566): breast label = most severe image label (Malignant > Benign > Normal),
               LE BI-RADS and DES BI-RADS (max over that energy's images), N1 change labels, density,
               post-operative flag, image paths per slot (LE_CC, LE_MLO, DES_CC, DES_MLO)
  folds.csv    patient-grouped, label-stratified 5-fold split (seed 0), one row per patient

Why the breast label rule: in 50/566 breasts the image labels differ, almost always LE = Benign while
DES = Normal (a benign finding visible only on the low-energy image), so labels describe what each image
shows, and the breast carries the worst one.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
# CDD-CESM: the dataset authors' repository (data/*.csv, reports/reports.csv) with the TCIA images in
# data/images/All/. Set CDDCESM_ROOT to its location, or place it at ../CDD-CESM-Dataset_original/CDD-CESM-Dataset.
REPO = Path(os.environ.get("CDDCESM_ROOT", HERE.parent / "CDD-CESM-Dataset_original" / "CDD-CESM-Dataset"))
OUT = HERE / "tables"
LABEL = "Pathology Classification/ Follow up"
SEVERITY = {"Normal": 0, "Benign": 1, "Malignant": 2}
POSTOP = r"post.?operative|operated|operative scar|lumpectomy|mastectomy|conservative surgery"


def max_birads(v):
    parts = [int(p) for p in re.findall(r"\d", str(v))]
    return max(parts) if parts else np.nan


def main():
    OUT.mkdir(exist_ok=True)
    lab = pd.concat([pd.read_csv(REPO / "data" / f) for f in ("train_set_All.csv", "test_set_All.csv")])
    rep = pd.read_csv(REPO / "reports" / "reports.csv")
    for d in (lab, rep):
        d["Image_name"] = d["Image_name"].str.strip()
    img = lab.merge(rep, on="Image_name", how="left", validate="one_to_one")
    assert len(img) == 2006 and img["report"].notna().all()

    img["energy"] = img["Type"].map({"DM": "LE", "CESM": "DES"})
    img["patient"] = img["Patient_ID"].astype(int)
    img["breast"] = img["patient"].astype(str) + "_" + img["Side"]
    img["label"] = img[LABEL].str.strip()
    img["birads"] = img["BIRADS"].map(max_birads)
    img["density"] = img["Breast density (ACR)"].astype(str).str.strip().replace({"_": np.nan, "nan": np.nan})
    img["path"] = [str(REPO / "data" / "images" / "All" / f"{n}.jpg") for n in img["Image_name"]]
    assert all(Path(p).exists() for p in img["path"]), "missing image files"
    img["text"] = (img["report"].fillna("") + " " + img["opinion"].fillna("")).str.lower().str.strip()
    img["findings"] = img["Findings"].fillna("").str.strip()
    img["slot"] = img["energy"] + "_" + img["View"]
    cols = ["Image_name", "path", "patient", "breast", "Side", "energy", "View", "slot", "label", "birads",
            "density", "Machine", "Age", "findings", "text"]
    img[cols].to_csv(OUT / "images.csv", index=False)

    rows = []
    for b, d in img.groupby("breast"):
        le, des = d[d.energy == "LE"], d[d.energy == "DES"]
        r = {"breast": b, "patient": d.patient.iloc[0], "side": d.Side.iloc[0],
             "label": max(d.label, key=SEVERITY.get),
             "label_le": max(le.label, key=SEVERITY.get), "label_des": max(des.label, key=SEVERITY.get),
             "birads_le": le.birads.max(), "birads_des": des.birads.max(),
             "density": le.density.dropna().mode().iloc[0] if le.density.notna().any() else np.nan,
             "machine": d.Machine.iloc[0], "age": d.Age.iloc[0],
             "postop": int(d.text.str.contains(POSTOP).any() or d.findings.str.lower().str.contains(POSTOP).any()),
             "n_images": len(d)}
        for s in ("LE_CC", "LE_MLO", "DES_CC", "DES_MLO"):
            p = d.loc[d.slot == s, "path"]
            r[s] = p.iloc[0] if len(p) else ""      # 2 breasts have 6 images; keep the first per slot
        rows.append(r)
    br = pd.DataFrame(rows)
    br["n1_change"] = (br.birads_des != br.birads_le).astype(int)
    br["n1_direction"] = np.sign(br.birads_des - br.birads_le).map({1: "up", -1: "down", 0: "same"})
    br.to_csv(OUT / "breasts.csv", index=False)

    # patient-grouped, stratified by the patient's worst breast label
    pat = br.groupby("patient")["label"].agg(lambda s: max(s, key=SEVERITY.get)).reset_index()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    pat["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(pat, pat.label, groups=pat.patient)):
        pat.loc[pat.index[te], "fold"] = k
    pat[["patient", "fold"]].to_csv(OUT / "folds.csv", index=False)

    br = br.merge(pat[["patient", "fold"]], on="patient")
    print(f"images {len(img)}, breasts {len(br)}, patients {br.patient.nunique()}")
    print("breast labels:", br.label.value_counts().to_dict())
    print("breasts with disagreeing image labels:", (img.groupby('breast').label.nunique() > 1).sum())
    print("N1: changed", br.n1_change.sum(), br.n1_direction.value_counts().to_dict())
    print("density (breasts):", br.density.value_counts(dropna=False).to_dict())
    print("post-operative breasts:", br.postop.sum(), br[br.postop == 1].label.value_counts().to_dict())
    print("slots missing:", {s: int((br[s] == "").sum()) for s in ("LE_CC", "LE_MLO", "DES_CC", "DES_MLO")})
    print("per fold (breasts by label):")
    print(pd.crosstab(br.fold, br.label).to_string())


if __name__ == "__main__":
    main()
