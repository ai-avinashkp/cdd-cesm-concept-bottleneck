# Concept extraction vs reference labels: test (50 breasts)

- **v2, all 36 concepts:** micro precision 0.995, recall 0.991, F1 0.993
- on the 20 concepts shared with the pilot: pilot v1 F1 0.901 (P 0.948, R 0.858) vs v2 F1 0.992 (P 0.992, R 0.992)

| concept | n_ref | TP | FP | FN | precision | recall | F1 | pilot F1 |
|---|---|---|---|---|---|---|---|---|
| le_mass | 22 | 22 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_shape_oval_round | 7 | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_shape_irregular | 15 | 15 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_margin_circumscribed | 7 | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_margin_indistinct | 9 | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_margin_spiculated | 7 | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_high_density | 16 | 15 | 0 | 1 | 1.00 | 0.94 | 0.97 |  |
| le_calc_suspicious | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.75 |
| le_calc_benign | 10 | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_distortion | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_asymmetry | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_skin_nipple | 8 | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.55 |
| le_edema | 10 | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_imln | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.80 |
| le_duct | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_postop | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_mass_enh | 17 | 17 | 1 | 0 | 0.94 | 1.00 | 0.97 | 0.87 |
| des_nme | 6 | 5 | 0 | 1 | 1.00 | 0.83 | 0.91 | 0.91 |
| des_homogeneous | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_heterogeneous | 17 | 17 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.97 |
| des_rim | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_faint | 0 | 0 | 0 | 0 | nan | nan | nan |  |
| des_focus_foci | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.86 |
| des_stippled | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_clumped | 0 | 0 | 0 | 0 | nan | nan | nan | nan |
| des_dist_linear_ductal | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.00 |
| des_dist_segmental | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_dist_regional_diffuse | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_non_enhancing | 8 | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.22 |
| des_shape_oval_round | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_shape_irregular | 15 | 15 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_margin_circumscribed | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_margin_irregular | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_margin_spiculated | 7 | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_septations | 0 | 0 | 0 | 0 | nan | nan | nan |  |
| des_satellites | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |

## v2 disagreements (breast, concept, extractor, reference)
- 12_L des_nme: extractor 0, reference 1
- 317_L le_high_density: extractor 0, reference 1
- 191_R des_mass_enh: extractor 1, reference 0