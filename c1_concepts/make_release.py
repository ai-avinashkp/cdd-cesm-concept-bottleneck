"""G5: build the release package of CEM-lexicon concept labels for CDD-CESM (derived labels only; no images or
report text). Output folder: release_concept_labels/ (README, image- and breast-level CSVs with excluded cells
left blank, guideline, validation summary, exclusion log).
"""
import shutil
from pathlib import Path

import pandas as pd

from concepts import DES, LE

HERE = Path(__file__).resolve().parent
OUT = HERE / "release_concept_labels"
COLS = [f"le_{c}" for c in LE] + [f"des_{c}" for c in DES]


def blank_excluded(t, m, key):
    t, m = t.set_index(key), m.set_index(key)
    return t[COLS].where(m[COLS] == 1).astype("Int64").reset_index()


def main():
    OUT.mkdir(exist_ok=True)
    ti, mi = pd.read_csv(HERE / "concepts_image.csv"), pd.read_csv(HERE / "concept_mask_image.csv")
    img = blank_excluded(ti, mi, "Image_name")      # other-energy columns are masked too, so they become blank
    tb, mb = pd.read_csv(HERE / "concepts_breast.csv"), pd.read_csv(HERE / "concept_mask_breast.csv")
    brs = blank_excluded(tb, mb, "breast")
    img.to_csv(OUT / "cddcesm_concepts_image.csv", index=False)
    brs.to_csv(OUT / "cddcesm_concepts_breast.csv", index=False)
    pd.read_csv(HERE / "excluded_cells.csv").to_csv(OUT / "excluded_cells.csv", index=False)
    shutil.copy(HERE / "validation" / "GUIDELINE.md", OUT / "GUIDELINE.md")
    prev = brs[COLS].sum().astype(int)
    n_blank_b = int(brs[COLS].isna().sum().sum())
    readme = f"""# CEM-lexicon concept labels for CDD-CESM (v2.2, {pd.Timestamp.today():%Y-%m-%d})

Derived, per-image and per-breast labels for **{len(COLS)} radiology concepts** (16 low-energy / BI-RADS
mammography, 19 recombined / ACR CEM lexicon) for the public **CDD-CESM** dataset (2,006 images, 566 breasts,
326 patients). Images and reports are NOT included; obtain them from TCIA.

**Source data (please cite):** Khaled R. et al., "Categorized contrast enhanced mammography dataset for
diagnostic and artificial intelligence research", *Scientific Data* 9:122 (2022); TCIA, DOI
10.7937/29kw-ae92 (CC BY 4.0).

## Files
| File | Content |
|---|---|
| `cddcesm_concepts_image.csv` | one row per image (`Image_name` as in CDD-CESM); 0/1 per concept for the image's own energy (`le_*` for low-energy images, `des_*` for recombined); blank = not applicable (other energy) or **excluded** |
| `cddcesm_concepts_breast.csv` | one row per breast (`<patient>_<L/R>`); a concept is 1 if any image of that energy states it; blank = excluded |
| `excluded_cells.csv` | every excluded cell and why ({len(pd.read_csv(OUT / 'excluded_cells.csv'))} image cells; {n_blank_b} breast cells) |
| `GUIDELINE.md` | annotation rules (BI-RADS 5th ed. + ACR CEM lexicon 2022 mapping) |

## How the labels were made
- **Extraction:** rule-based extraction from the CDD-CESM radiology reports, with clause-level negation,
  typo normalisation and per-energy scoping. Code: `concepts.py`.
- **Validation:** against reference labels read from the reports following `GUIDELINE.md`.
  - The reference reader was an LLM (Claude), not a radiologist.
  - Dev set: 100 breasts.
  - Held-out test set: 50 breasts, disjoint patients, extractor frozen. **Micro-F1 0.993** (3 errors in
    1,800 cells; all fixed afterwards).
  - Independent check against the dataset's own radiologist-written `Findings` column: 97.5% of concepts
    stated there are present in the reference labels. All mismatches are report-vs-Findings source
    differences.
- **Exclusions:** a cell is excluded (blank) where the view-specific Findings column states a concept the
  report does not (a source conflict), or where a guideline decision needs a radiologist's confirmation.

## Caveats
- Labels describe what the radiologists *wrote*, not an independent read of the images.
- The reference standard was produced by an LLM reader. A radiologist re-read of a sample is recommended
  before clinical use.
- Report text is shared by the CC and MLO views; a concept may not be visible in both views.

## Breast-level prevalence (number of breasts with the concept)
""" + "\n".join(f"- `{c}`: {v}" for c, v in prev.items()) + """

## Licence
CC BY 4.0, matching CDD-CESM. Cite the CDD-CESM dataset and the accompanying paper.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"release written: {sorted(p.name for p in OUT.iterdir())}; image rows {len(img)}, breast rows {len(brs)}")


if __name__ == "__main__":
    main()
