# Concept extraction vs reference labels: dev (100 breasts)

- **v2, all 35 concepts:** micro precision 1.000, recall 1.000, F1 1.000
- on the 20 concepts shared with the pilot: pilot v1 F1 0.926 (P 0.938, R 0.914) vs v2 F1 1.000 (P 1.000, R 1.000)

| concept | n_ref | TP | FP | FN | precision | recall | F1 | pilot F1 |
|---|---|---|---|---|---|---|---|---|
| le_mass | 35 | 35 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_shape_oval_round | 15 | 15 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_shape_irregular | 20 | 20 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_margin_circumscribed | 7 | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_margin_indistinct | 15 | 15 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_margin_spiculated | 13 | 13 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_high_density | 26 | 26 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_calc_suspicious | 8 | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.86 |
| le_calc_benign | 18 | 18 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.91 |
| le_distortion | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_asymmetry | 16 | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_skin_nipple | 9 | 9 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.88 |
| le_edema | 25 | 25 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| le_imln | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.86 |
| le_duct | 5 | 5 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| le_postop | 10 | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_mass_enh | 35 | 35 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.86 |
| des_nme | 14 | 14 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_homogeneous | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_heterogeneous | 27 | 27 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.98 |
| des_rim | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_faint | 7 | 7 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_focus_foci | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.80 |
| des_stippled | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_clumped | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_dist_linear_ductal | 6 | 6 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.50 |
| des_dist_segmental | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_dist_regional_diffuse | 4 | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| des_non_enhancing | 8 | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.00 |
| des_shape_oval_round | 11 | 11 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_irregular | 28 | 28 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_margin_circumscribed | 6 | 6 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_margin_spiculated | 16 | 16 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_septations | 2 | 2 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |
| des_satellites | 3 | 3 | 0 | 0 | 1.00 | 1.00 | 1.00 |  |

## v2 disagreements (breast, concept, extractor, reference)