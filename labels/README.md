# CEM-lexicon concept labels for CDD-CESM (v2.2, 2026-11-17)

Derived, per-image and per-breast labels for **35 radiology concepts** (16 low-energy / BI-RADS
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
| `excluded_cells.csv` | every excluded cell and why (35 image cells; 25 breast cells) |
| `GUIDELINE.md` | annotation rules (BI-RADS 5th ed. + ACR CEM lexicon 2022 mapping) |

## How the labels were made
- **Extraction:** rule-based extraction from the CDD-CESM radiology reports, with clause-level negation,
  typo normalisation and per-energy scoping. Code: `concepts.py`.
- **Validation:** against reference labels read from the reports following `GUIDELINE.md`.
  - The reference reader was an LLM (Claude), not a radiologist. An author re-read 20 of the sampled breasts
    by hand and agreed with all of their reference labels.
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
- `le_mass`: 162
- `le_shape_oval_round`: 77
- `le_shape_irregular`: 88
- `le_margin_circumscribed`: 43
- `le_margin_indistinct`: 72
- `le_margin_spiculated`: 56
- `le_high_density`: 101
- `le_calc_suspicious`: 43
- `le_calc_benign`: 86
- `le_distortion`: 30
- `le_asymmetry`: 65
- `le_skin_nipple`: 52
- `le_edema`: 92
- `le_imln`: 23
- `le_duct`: 12
- `le_postop`: 47
- `des_mass_enh`: 150
- `des_nme`: 67
- `des_homogeneous`: 18
- `des_heterogeneous`: 138
- `des_rim`: 19
- `des_faint`: 18
- `des_focus_foci`: 25
- `des_stippled`: 11
- `des_clumped`: 7
- `des_dist_linear_ductal`: 25
- `des_dist_segmental`: 22
- `des_dist_regional_diffuse`: 16
- `des_non_enhancing`: 60
- `des_shape_oval_round`: 36
- `des_irregular`: 127
- `des_margin_circumscribed`: 24
- `des_margin_spiculated`: 59
- `des_septations`: 6
- `des_satellites`: 17

## Licence
CC BY 4.0, matching CDD-CESM. Cite the CDD-CESM dataset and the accompanying paper.
