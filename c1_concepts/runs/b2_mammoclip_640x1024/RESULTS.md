# Baseline B0 results (EfficientNet-B0, 512x768, 5 folds, 566 breasts)

## 1. Concept prediction (breast level)

| concept | positives | AUC |
|---|---|---|
| le_mass | 162 | 0.895 |
| le_shape_oval_round | 77 | 0.844 |
| le_shape_irregular | 88 | 0.931 |
| le_margin_circumscribed | 43 | 0.838 |
| le_margin_indistinct | 72 | 0.845 |
| le_margin_spiculated | 56 | 0.935 |
| le_high_density | 101 | 0.944 |
| le_calc_suspicious | 43 | 0.908 |
| le_calc_benign | 86 | 0.871 |
| le_distortion | 30 | 0.735 |
| le_asymmetry | 65 | 0.657 |
| le_skin_nipple | 52 | 0.923 |
| le_edema | 92 | 0.979 |
| le_imln | 23 | 0.829 |
| le_duct | 12 | 0.699 |
| le_postop | 47 | 0.789 |
| des_mass_enh | 150 | 0.931 |
| des_nme | 67 | 0.818 |
| des_homogeneous | 18 | 0.753 |
| des_heterogeneous | 138 | 0.934 |
| des_rim | 19 | 0.709 |
| des_faint | 18 | 0.546 |
| des_focus_foci | 25 | 0.718 |
| des_stippled | 11 | 0.733 |
| des_clumped | 7 | 0.631 |
| des_dist_linear_ductal | 25 | 0.653 |
| des_dist_segmental | 22 | 0.842 |
| des_dist_regional_diffuse | 16 | 0.916 |
| des_non_enhancing | 60 | 0.686 |
| des_shape_oval_round | 36 | 0.812 |
| des_irregular | 127 | 0.925 |
| des_margin_circumscribed | 24 | 0.771 |
| des_margin_spiculated | 59 | 0.896 |
| des_septations | 6 | 0.619 |
| des_satellites | 17 | 0.897 |

Mean concept AUC: 0.812 over 35 concepts with >= 5 positives

## 2–4. Diagnosis (breast level, out-of-fold)

| model | 3-class accuracy | 3-class macro AUC | benign vs malignant AUC |
|---|---|---|---|
| black box (label head) | 0.698 | 0.849 | 0.866 |
| concept bottleneck (predicted concepts) | 0.701 | 0.867 | 0.886 |
|   + radiologist corrects 1 most uncertain concept(s) | 0.753 | 0.882 | 0.885 |
|   + radiologist corrects 3 most uncertain concept(s) | 0.767 | 0.894 | 0.892 |
|   + radiologist corrects 5 most uncertain concept(s) | 0.758 | 0.898 | 0.893 |
|   + radiologist corrects all most uncertain concept(s) | 0.788 | 0.921 | 0.928 |
| upper bound: head trained on report concepts | 0.898 | 0.966 | 0.947 |