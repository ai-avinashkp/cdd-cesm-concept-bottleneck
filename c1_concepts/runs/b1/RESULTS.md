# Baseline B0 results (EfficientNet-B0, 512x768, 5 folds, 566 breasts)

## 1. Concept prediction (breast level)

| concept | positives | AUC |
|---|---|---|
| le_mass | 162 | 0.876 |
| le_shape_oval_round | 77 | 0.788 |
| le_shape_irregular | 88 | 0.936 |
| le_margin_circumscribed | 43 | 0.763 |
| le_margin_indistinct | 72 | 0.812 |
| le_margin_spiculated | 56 | 0.946 |
| le_high_density | 101 | 0.944 |
| le_calc_suspicious | 43 | 0.834 |
| le_calc_benign | 86 | 0.769 |
| le_distortion | 30 | 0.730 |
| le_asymmetry | 65 | 0.653 |
| le_skin_nipple | 52 | 0.917 |
| le_edema | 92 | 0.985 |
| le_imln | 23 | 0.672 |
| le_duct | 12 | 0.517 |
| le_postop | 47 | 0.807 |
| des_mass_enh | 150 | 0.936 |
| des_nme | 67 | 0.803 |
| des_homogeneous | 18 | 0.731 |
| des_heterogeneous | 138 | 0.936 |
| des_rim | 19 | 0.729 |
| des_faint | 18 | 0.540 |
| des_focus_foci | 25 | 0.680 |
| des_stippled | 11 | 0.683 |
| des_clumped | 7 | 0.656 |
| des_dist_linear_ductal | 25 | 0.687 |
| des_dist_segmental | 22 | 0.762 |
| des_dist_regional_diffuse | 16 | 0.932 |
| des_non_enhancing | 60 | 0.689 |
| des_shape_oval_round | 36 | 0.776 |
| des_irregular | 127 | 0.940 |
| des_margin_circumscribed | 24 | 0.756 |
| des_margin_spiculated | 59 | 0.934 |
| des_septations | 6 | 0.503 |
| des_satellites | 17 | 0.854 |

Mean concept AUC: 0.785 over 35 concepts with >= 5 positives

## 2–4. Diagnosis (breast level, out-of-fold)

| model | 3-class accuracy | 3-class macro AUC | benign vs malignant AUC |
|---|---|---|---|
| black box (label head) | 0.655 | 0.829 | 0.843 |
| concept bottleneck (predicted concepts) | 0.680 | 0.858 | 0.889 |
|   + radiologist corrects 1 most uncertain concept(s) | 0.731 | 0.886 | 0.885 |
|   + radiologist corrects 3 most uncertain concept(s) | 0.770 | 0.905 | 0.897 |
|   + radiologist corrects 5 most uncertain concept(s) | 0.779 | 0.910 | 0.898 |
|   + radiologist corrects all most uncertain concept(s) | 0.753 | 0.896 | 0.924 |
| upper bound: head trained on report concepts | 0.898 | 0.966 | 0.947 |