# G4 / cross-energy concept bottleneck pilot (Mammo-CLIP B2, LE-only models)

## Q1. Predicting the contrast (DES) findings

| DES concept | positives | morphology baseline (LE report concepts) | **LE image (cross-energy)** | DES image (reference) |
|---|---|---|---|---|
| des_mass_enh | 150 | 0.941 | 0.892 | 0.931 |
| des_nme | 67 | 0.842 | 0.778 | 0.818 |
| des_homogeneous | 18 | 0.766 | 0.685 | 0.753 |
| des_heterogeneous | 138 | 0.946 | 0.884 | 0.934 |
| des_rim | 19 | 0.612 | 0.671 | 0.709 |
| des_faint | 18 | 0.683 | 0.403 | 0.546 |
| des_focus_foci | 25 | 0.445 | 0.569 | 0.718 |
| des_stippled | 11 | 0.625 | 0.648 | 0.733 |
| des_dist_linear_ductal | 25 | 0.681 | 0.723 | 0.653 |
| des_dist_segmental | 22 | 0.756 | 0.757 | 0.842 |
| des_dist_regional_diffuse | 16 | 0.895 | 0.887 | 0.916 |
| des_non_enhancing | 60 | 0.914 | 0.754 | 0.686 |
| des_shape_oval_round | 36 | 0.958 | 0.779 | 0.812 |
| des_irregular | 127 | 0.946 | 0.871 | 0.925 |
| des_margin_circumscribed | 24 | 0.949 | 0.781 | 0.771 |
| des_margin_spiculated | 59 | 0.958 | 0.901 | 0.896 |
| des_satellites | 17 | 0.764 | 0.778 | 0.897 |
| **mean** | | 0.805 | 0.751 | 0.797 |

## Q2. Iodine-trace test: enhancement among LE-mass breasts

- breasts with an LE mass: 162; enhancing 128, non-enhancing 34
- morphology baseline (LE report concepts): AUC 0.847
- **LE image (cross-energy model): AUC 0.824**
- DES image (reference): AUC 0.939
- Reading: if the LE image is far above the morphology baseline and close to the DES reference, contrast information is probably leaking into LE (risk B2); if it is close to the morphology baseline, the LE model is using morphology, as intended.

## Q3. LE-only diagnosis: does learning the contrast concepts help?

| model | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| LE black box (cross-energy model) | 0.677 | 0.834 | 0.840 |
| CE-CBM: LE + predicted DES concepts (cross) | 0.670 | 0.826 | 0.848 |
| CBM: LE concepts only (cross model) | 0.673 | 0.829 | 0.850 |
| LE black box (standard LE model) | 0.671 | 0.837 | 0.842 |
| CBM: LE concepts only (standard LE model) | 0.671 | 0.835 | 0.855 |
| reference: LE+DES black box (full model) | 0.698 | 0.849 | 0.866 |

Paired patient bootstrap, CE-CBM minus LE-only CBM (standard model): accuracy -0.002 [-0.021, +0.017], P(<=0)=0.602; macro AUC -0.010 [-0.018, -0.001], P(<=0)=0.993; B-vs-M AUC -0.007 [-0.019, +0.006], P(<=0)=0.865