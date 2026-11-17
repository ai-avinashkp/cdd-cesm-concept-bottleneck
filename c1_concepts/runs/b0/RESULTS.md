# Baseline B0 results (EfficientNet-B0, 512x768, 5 folds, 566 breasts)

## 1. Concept prediction (breast level)

| concept | positives | AUC |
|---|---|---|
| le_mass | 162 | 0.876 |
| le_shape_oval_round | 77 | 0.788 |
| le_shape_irregular | 88 | 0.932 |
| le_margin_circumscribed | 43 | 0.756 |
| le_margin_indistinct | 72 | 0.808 |
| le_margin_spiculated | 56 | 0.942 |
| le_high_density | 101 | 0.947 |
| le_calc_suspicious | 43 | 0.862 |
| le_calc_benign | 86 | 0.727 |
| le_distortion | 30 | 0.718 |
| le_asymmetry | 65 | 0.663 |
| le_skin_nipple | 52 | 0.898 |
| le_edema | 92 | 0.977 |
| le_imln | 23 | 0.734 |
| le_duct | 12 | 0.476 |
| le_postop | 47 | 0.797 |
| des_mass_enh | 150 | 0.924 |
| des_nme | 67 | 0.827 |
| des_homogeneous | 18 | 0.716 |
| des_heterogeneous | 138 | 0.932 |
| des_rim | 19 | 0.685 |
| des_faint | 18 | 0.515 |
| des_focus_foci | 25 | 0.685 |
| des_stippled | 11 | 0.616 |
| des_clumped | 7 | 0.677 |
| des_dist_linear_ductal | 25 | 0.732 |
| des_dist_segmental | 22 | 0.788 |
| des_dist_regional_diffuse | 16 | 0.941 |
| des_non_enhancing | 60 | 0.637 |
| des_shape_oval_round | 36 | 0.736 |
| des_shape_irregular | 118 | 0.928 |
| des_margin_circumscribed | 24 | 0.678 |
| des_margin_irregular | 9 | 0.838 |
| des_margin_spiculated | 59 | 0.924 |
| des_septations | 6 | 0.353 |
| des_satellites | 17 | 0.864 |

Mean concept AUC: 0.775 over 36 concepts with >= 5 positives

## 2–4. Diagnosis (breast level, out-of-fold)

| model | 3-class accuracy | 3-class macro AUC | benign vs malignant AUC |
|---|---|---|---|
| black box (label head) | 0.666 | 0.829 | 0.844 |
| concept bottleneck (predicted concepts) | 0.678 | 0.855 | 0.881 |
|   + radiologist corrects 1 most uncertain concept(s) | 0.746 | 0.887 | 0.888 |
|   + radiologist corrects 3 most uncertain concept(s) | 0.765 | 0.914 | 0.900 |
|   + radiologist corrects 5 most uncertain concept(s) | 0.781 | 0.924 | 0.911 |
|   + radiologist corrects all most uncertain concept(s) | 0.797 | 0.921 | 0.938 |
| upper bound: head trained on report concepts | 0.901 | 0.967 | 0.948 |