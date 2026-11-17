"""C1 concept targets v2.2 with exclusion masks (user decision 2026-10-31: exclude conflicting labels,
keep the clean data).

Targets come from the report text (extractor v2.2). A concept cell is EXCLUDED (mask = 0: not trained on,
not scored) when:
  (a) source conflict: the radiologists' view-specific Findings column states the concept for an image but the
      report text does not (report and Findings disagree);
  (b) a guideline call that a radiologist should confirm (MANUAL below).
Excluding a cell at image level also excludes it at breast level for that breast.

Writes:
  concepts_breast.csv      breast-level targets (0/1)
  concept_mask_breast.csv  breast-level mask (1 = usable, 0 = excluded)
  concepts_image.csv / concept_mask_image.csv   image-level targets/mask (own energy only; other energy = 0)
  excluded_cells.csv       every excluded cell with its reason
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from concepts import DES, LE, extract  # noqa: E402

TAB = HERE.parent / "pipeline" / "tables"
COLS = [f"le_{c}" for c in LE] + [f"des_{c}" for c in DES]
MANUAL = {  # (breast, concept): reason (see validation/SPOT_CHECK_RESULT.md)
    ("187_R", "le_mass"): "guideline call: 'ill defined lesion' not counted as mass",
    ("152_L", "le_calc_suspicious"): "guideline call: 'scattered calcifications' in a malignant mass",
    ("152_L", "le_calc_benign"): "guideline call: 'scattered calcifications' in a malignant mass",
}


def main():
    img = pd.read_csv(TAB / "images.csv")
    ti = pd.DataFrame(0, index=img.Image_name, columns=COLS)
    mi = pd.DataFrame(0, index=img.Image_name, columns=COLS)
    log = []
    for _, r in img.iterrows():
        pre, lex = ("le_", LE) if r.energy == "LE" else ("des_", DES)
        rep = extract(r.text, r.energy)
        fnd = extract(str(r.findings if isinstance(r.findings, str) else "").replace("$", ". "), r.energy)
        for c in lex:
            col = pre + c
            ti.at[r.Image_name, col] = rep[c]
            mi.at[r.Image_name, col] = 1
            if fnd[c] == 1 and rep[c] == 0:
                mi.at[r.Image_name, col] = 0
                log.append({"breast": r.breast, "image": r.Image_name, "concept": col,
                            "reason": f"source conflict: Findings states it, report does not ('{r.findings}')"})
    for (b, c), why in MANUAL.items():
        for name in img.loc[img.breast == b, "Image_name"]:
            if mi.at[name, c] == 1:
                mi.at[name, c] = 0
                log.append({"breast": b, "image": name, "concept": c, "reason": why})

    tb, mb = [], []
    for b, d in img.groupby("breast"):
        names = d.Image_name
        t, m = ti.loc[names], mi.loc[names]
        row_t, row_m = {"breast": b}, {"breast": b}
        for c in COLS:
            energy_imgs = d.loc[d.energy == ("LE" if c.startswith("le_") else "DES"), "Image_name"]
            row_t[c] = int(t.loc[energy_imgs, c].max()) if len(energy_imgs) else 0
            row_m[c] = int(m.loc[energy_imgs, c].min()) if len(energy_imgs) else 0
        tb.append(row_t)
        mb.append(row_m)
    tb, mb = pd.DataFrame(tb), pd.DataFrame(mb)
    tb.to_csv(HERE / "concepts_breast.csv", index=False)
    mb.to_csv(HERE / "concept_mask_breast.csv", index=False)
    ti.reset_index().to_csv(HERE / "concepts_image.csv", index=False)
    mi.reset_index().to_csv(HERE / "concept_mask_image.csv", index=False)
    lg = pd.DataFrame(log)
    lg.to_csv(HERE / "excluded_cells.csv", index=False)

    excl = (mb[COLS] == 0).sum()
    print(f"concepts {len(COLS)}; breasts {len(tb)}")
    print(f"excluded image cells: {len(lg)}; breast cells excluded: {int(excl.sum())} of {len(tb) * len(COLS)} "
          f"({excl.sum() / (len(tb) * len(COLS)):.2%}); breasts touched: {int((mb[COLS] == 0).any(axis=1).sum())}")
    print("excluded breast cells per concept:", {c: int(v) for c, v in excl.items() if v})


if __name__ == "__main__":
    main()
