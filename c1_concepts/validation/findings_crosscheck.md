# Cross-check against the dataset's 'Findings' column (independent radiologist summary)

### A. Reference labels (LLM reader), 150 breasts

Overall: 504/517 Findings-stated concepts are in the reference (97.5%).

| concept | Findings positive | also in reference | agreement |
|---|---|---|---|
| le_mass | 57 | 57 | 1.00 |
| le_shape_oval_round | 3 | 3 | 1.00 |
| le_shape_irregular | 1 | 1 | 1.00 |
| le_margin_circumscribed | 12 | 12 | 1.00 |
| le_margin_indistinct | 24 | 24 | 1.00 |
| le_margin_spiculated | 19 | 19 | 1.00 |
| le_high_density | 42 | 42 | 1.00 |
| le_calc_suspicious | 12 | 12 | 1.00 |
| le_calc_benign | 28 | 27 | 0.96 |
| le_distortion | 7 | 6 | 0.86 |
| le_asymmetry | 21 | 20 | 0.95 |
| le_skin_nipple | 11 | 11 | 1.00 |
| le_edema | 36 | 35 | 0.97 |
| le_imln | 7 | 6 | 0.86 |
| le_duct | 6 | 6 | 1.00 |
| le_postop | 13 | 13 | 1.00 |
| des_mass_enh | 48 | 48 | 1.00 |
| des_nme | 20 | 20 | 1.00 |
| des_homogeneous | 4 | 4 | 1.00 |
| des_heterogeneous | 42 | 42 | 1.00 |
| des_rim | 5 | 5 | 1.00 |
| des_faint | 9 | 7 | 0.78 |
| des_focus_foci | 7 | 7 | 1.00 |
| des_stippled | 5 | 5 | 1.00 |
| des_clumped | 2 | 2 | 1.00 |
| des_dist_linear_ductal | 7 | 7 | 1.00 |
| des_dist_segmental | 6 | 6 | 1.00 |
| des_dist_regional_diffuse | 6 | 6 | 1.00 |
| des_non_enhancing | 14 | 14 | 1.00 |
| des_shape_oval_round | 1 | 1 | 1.00 |
| des_shape_irregular | 8 | 8 | 1.00 |
| des_margin_circumscribed | 4 | 3 | 0.75 |
| des_margin_irregular | 5 | 0 | 0.00 |
| des_margin_spiculated | 18 | 18 | 1.00 |
| des_septations | 2 | 2 | 1.00 |
| des_satellites | 5 | 5 | 1.00 |

Cases where Findings states a concept that the reference labels lack:
- 128_L le_calc_benign (Findings: Asymmetrical increased density and equal density nodule with indistinct margin and overlying linear branching macrocalcifications | Faint focal enhancement)
- 280_L le_distortion (Findings: Architectural distortion with overlying microcalcifications and nipple retraction | Multiple irregular heterogenously enhancing masses | Multiple high density masses with indistinct margin and overlying microcalcifications and nipple retraction)
- 265_R le_asymmetry (Findings: Heterogenously enhancing mass with irregular margin$Stippled enhancement | Asymmetrical increased density | High density mass with microlobulated margin)
- 55_R le_edema (Findings: Edema and postoperative scar with two focal asymmetrical increased densities and nipple retraction | Two heterogenously enhancing irregular masses)
- 291_R le_imln (Findings: Two high density masses with speculated margin and intraductal extension and nipple retraction$IMLN | Two irregular heterogenously enhancing masses with intraductal extension and surrounding satellites and nipple retraction | Two high density masses with speculated margin and intraductal extension and nipple retraction)
- 181_L des_faint (Findings: Faint patchy areas of heterogenous enhancement | Asymmetrical increased density and IMLN)
- 94_R des_faint (Findings: Homogenously enhancing mass with speculated margin | Faintly enhancing mass with speculated margin | Asymmetrical increased density)
- 174_L des_margin_circumscribed (Findings: Enhancing mass with circumscribed margin | High density mass with microlobulated margin$Benign macrocalcification | Irregular enhancing mass)
- 265_R des_margin_irregular (Findings: Heterogenously enhancing mass with irregular margin$Stippled enhancement | Asymmetrical increased density | High density mass with microlobulated margin)
- 168_L des_margin_irregular (Findings: Multiple heterogenously enhancing masses with irregular margin | Multiple high density masses with microlobulated margin and edema pattern)
- 209_L des_margin_irregular (Findings: Heterogenously enhancing mass with irregular margin | High density mass with microlobulated margin and edema pattern)
- 315_L des_margin_irregular (Findings: High density mass with indistinct margin and two equal density nodules | Heterogenously enhancing mass with irregular margin)
- 173_L des_margin_irregular (Findings: Flap with high density mass with microlobulated margin | Flap | Heterogenously enhancing mass with irregular margin)

### B. Extractor v2.1 on report text, all 566 breasts

Overall: 1491/1521 Findings-stated concepts are in the reference (98.0%).

| concept | Findings positive | also in reference | agreement |
|---|---|---|---|
| le_mass | 161 | 160 | 0.99 |
| le_shape_oval_round | 8 | 8 | 1.00 |
| le_shape_irregular | 4 | 4 | 1.00 |
| le_margin_circumscribed | 35 | 35 | 1.00 |
| le_margin_indistinct | 67 | 65 | 0.97 |
| le_margin_spiculated | 54 | 54 | 1.00 |
| le_high_density | 101 | 101 | 1.00 |
| le_calc_suspicious | 43 | 42 | 0.98 |
| le_calc_benign | 87 | 85 | 0.98 |
| le_distortion | 31 | 30 | 0.97 |
| le_asymmetry | 65 | 61 | 0.94 |
| le_skin_nipple | 22 | 22 | 1.00 |
| le_edema | 94 | 92 | 0.98 |
| le_imln | 23 | 22 | 0.96 |
| le_duct | 12 | 12 | 1.00 |
| le_postop | 45 | 45 | 1.00 |
| des_mass_enh | 141 | 141 | 1.00 |
| des_nme | 67 | 67 | 1.00 |
| des_homogeneous | 16 | 16 | 1.00 |
| des_heterogeneous | 132 | 131 | 0.99 |
| des_rim | 18 | 18 | 1.00 |
| des_faint | 20 | 18 | 0.90 |
| des_focus_foci | 26 | 25 | 0.96 |
| des_stippled | 11 | 11 | 1.00 |
| des_clumped | 7 | 7 | 1.00 |
| des_dist_linear_ductal | 25 | 25 | 1.00 |
| des_dist_segmental | 22 | 22 | 1.00 |
| des_dist_regional_diffuse | 16 | 16 | 1.00 |
| des_non_enhancing | 59 | 58 | 0.98 |
| des_shape_oval_round | 2 | 2 | 1.00 |
| des_shape_irregular | 24 | 24 | 1.00 |
| des_margin_circumscribed | 11 | 10 | 0.91 |
| des_margin_irregular | 9 | 1 | 0.11 |
| des_margin_spiculated | 40 | 38 | 0.95 |
| des_septations | 6 | 6 | 1.00 |
| des_satellites | 17 | 17 | 1.00 |

Cases (B): 30; first 25:
- 319_L le_mass (Findings: Circumscribed mixed iso and low density mass | Glandular enhancement)
- 146_R le_margin_indistinct (Findings: Multiple heterogenously enhancing masses with speculated margin | Multiple high density masses with indistinct margin | Multiple high density masses with speculated margin)
- 285_L le_margin_indistinct (Findings: Homogenously enhancing mass with circumscribed margin$ non enhancement | Focal asymmetry and two equal density mass with obscured margin | Focal asymmetry and two equal density masses with circumscribed margin)
- 162_R le_calc_suspicious (Findings: Linear non mass enhancement | Asymmetrical increased density and cluster of microcalcifications | High density mass with indistinct margin)
- 128_L le_calc_benign (Findings: Asymmetrical increased density and equal density nodule with indistinct margin and overlying linear branching macrocalcifications | Faint focal enhancement)
- 2_R le_calc_benign (Findings: Multiple equal density  masses$Benign macrocalcifications | Multiple equal density masses$Benign macrocalcifications | Multiple homogenously enhancing  masses)
- 280_L le_distortion (Findings: Architectural distortion with overlying microcalcifications and nipple retraction | Multiple irregular heterogenously enhancing masses | Multiple high density masses with indistinct margin and overlying microcalcifications and nipple retraction)
- 147_R le_asymmetry (Findings: Architectural distortion | Asymmetrical increased density | No mass or non mass enhancement)
- 162_R le_asymmetry (Findings: Linear non mass enhancement | Asymmetrical increased density and cluster of microcalcifications | High density mass with indistinct margin)
- 265_R le_asymmetry (Findings: Heterogenously enhancing mass with irregular margin$Stippled enhancement | Asymmetrical increased density | High density mass with microlobulated margin)
- 281_R le_asymmetry (Findings: Architectural distortion and equal density mass with indistinct margin | Asymmetrical increased density | Faint irregular heterogenously enhancing mass | Faint irregular heterogenously enhancing mass$ Homogenously enhancing mass)
- 241_L le_edema (Findings: Edema and postoperative scar$Benign macrocalcifications | Non enhancement)
- 55_R le_edema (Findings: Edema and postoperative scar with two focal asymmetrical increased densities and nipple retraction | Two heterogenously enhancing irregular masses)
- 291_R le_imln (Findings: Two high density masses with speculated margin and intraductal extension and nipple retraction$IMLN | Two irregular heterogenously enhancing masses with intraductal extension and surrounding satellites and nipple retraction | Two high density masses with speculated margin and intraductal extension and nipple retraction)
- 82_L des_heterogeneous (Findings: High density mass with indistinct margin | Multiple heterogenously enhancing  masses with speculated margin | High density mass with indistinct margin and two adjacent masses with indistinct margin)
- 181_L des_faint (Findings: Faint patchy areas of heterogenous enhancement | Asymmetrical increased density and IMLN)
- 94_R des_faint (Findings: Homogenously enhancing mass with speculated margin | Faintly enhancing mass with speculated margin | Asymmetrical increased density)
- 87_L des_focus_foci (Findings: Enhancing foci with ductal non mass enhancement | Two homogenously enhancing masses and another mass with irregular rim enhancement and ductal non mass enhancement | Asymmetrical increased density with overlying segmental microcalcification)
- 156_L des_non_enhancing (Findings: Non enhancement | Diffuse edematous changes)
- 174_L des_margin_circumscribed (Findings: Enhancing mass with circumscribed margin | High density mass with microlobulated margin$Benign macrocalcification | Irregular enhancing mass)
- 122_L des_margin_irregular (Findings: Three heterogenously enhancing masses with irregular margin and nipple retraction | Three heterogenously enhancing masses with speculated margin and nipple retraction | Three high density masses with speculated margin and nipple retraction)
- 168_L des_margin_irregular (Findings: Multiple heterogenously enhancing masses with irregular margin | Multiple high density masses with microlobulated margin and edema pattern)
- 173_L des_margin_irregular (Findings: Flap with high density mass with microlobulated margin | Flap | Heterogenously enhancing mass with irregular margin)
- 209_L des_margin_irregular (Findings: Heterogenously enhancing mass with irregular margin | High density mass with microlobulated margin and edema pattern)
- 265_R des_margin_irregular (Findings: Heterogenously enhancing mass with irregular margin$Stippled enhancement | Asymmetrical increased density | High density mass with microlobulated margin)