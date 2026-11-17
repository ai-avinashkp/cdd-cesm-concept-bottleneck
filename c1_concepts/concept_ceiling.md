# C1 concept ceiling (report concepts only, extractor v2.1)

- 566 breasts, 326 patients; shared patient-grouped 5 folds; bootstrap 95% CI over patients (2,000 resamples)
- This is an **upper bound**: concepts and labels come from the same radiologists (risk B4).

| concept set | n concepts | 3-class accuracy | 3-class macro AUC | benign vs malignant AUC |
|---|---|---|---|---|
| LE concepts only | 16 | 0.852 [0.822, 0.878] | 0.922 [0.901, 0.941] | 0.925 [0.898, 0.948] |
| DES concepts only | 19 | 0.818 [0.787, 0.849] | 0.889 [0.868, 0.910] | 0.943 [0.918, 0.965] |
| LE + DES concepts | 35 | 0.898 [0.873, 0.922] | 0.966 [0.952, 0.979] | 0.951 [0.927, 0.971] |

## Concept weights (benign vs malignant, all abnormal breasts; log-odds per concept)

| concept | weight | breasts with concept (abnormal) |
|---|---|---|
| des_non_enhancing | -1.45 | 60 |
| des_margin_circumscribed | -1.11 | 24 |
| le_calc_benign | -0.65 | 86 |
| des_septations | -0.60 | 6 |
| le_duct | -0.57 | 12 |
| le_margin_circumscribed | -0.42 | 43 |
| des_shape_oval_round | -0.37 | 36 |
| le_imln | -0.35 | 23 |
| le_calc_suspicious | +0.84 | 43 |
| le_edema | +0.92 | 91 |
| des_nme | +0.99 | 67 |
| le_skin_nipple | +0.99 | 51 |
| des_margin_spiculated | +0.99 | 59 |
| des_dist_linear_ductal | +1.23 | 25 |
| des_irregular | +1.25 | 127 |
| le_distortion | +1.29 | 29 |
| des_mass_enh | +1.41 | 150 |
| des_dist_segmental | +1.49 | 22 |

## Concept prevalence (all 566 breasts)

| concept | breasts | Normal | Benign | Malignant |
|---|---|---|---|---|
| le_mass | 162 | 0 | 55 | 107 |
| le_shape_oval_round | 77 | 0 | 52 | 25 |
| le_shape_irregular | 88 | 0 | 4 | 84 |
| le_margin_circumscribed | 43 | 0 | 35 | 8 |
| le_margin_indistinct | 72 | 0 | 25 | 47 |
| le_margin_spiculated | 56 | 0 | 1 | 55 |
| le_high_density | 101 | 0 | 10 | 91 |
| le_calc_suspicious | 43 | 0 | 5 | 38 |
| le_calc_benign | 86 | 0 | 66 | 20 |
| le_distortion | 30 | 1 | 3 | 26 |
| le_asymmetry | 65 | 5 | 21 | 39 |
| le_skin_nipple | 52 | 1 | 5 | 46 |
| le_edema | 92 | 1 | 28 | 63 |
| le_imln | 23 | 0 | 20 | 3 |
| le_duct | 12 | 0 | 10 | 2 |
| le_postop | 47 | 0 | 34 | 13 |
| des_mass_enh | 150 | 0 | 25 | 125 |
| des_nme | 67 | 0 | 14 | 53 |
| des_homogeneous | 18 | 0 | 10 | 8 |
| des_heterogeneous | 138 | 0 | 13 | 125 |
| des_rim | 19 | 0 | 7 | 12 |
| des_faint | 18 | 0 | 10 | 8 |
| des_focus_foci | 25 | 0 | 16 | 9 |
| des_stippled | 11 | 0 | 8 | 3 |
| des_clumped | 7 | 0 | 1 | 6 |
| des_dist_linear_ductal | 25 | 0 | 4 | 21 |
| des_dist_segmental | 22 | 0 | 2 | 20 |
| des_dist_regional_diffuse | 16 | 0 | 3 | 13 |
| des_non_enhancing | 60 | 0 | 59 | 1 |
| des_shape_oval_round | 36 | 0 | 22 | 14 |
| des_irregular | 127 | 0 | 6 | 121 |
| des_margin_circumscribed | 24 | 0 | 20 | 4 |
| des_margin_spiculated | 59 | 0 | 0 | 59 |
| des_septations | 6 | 0 | 6 | 0 |
| des_satellites | 17 | 0 | 0 | 17 |