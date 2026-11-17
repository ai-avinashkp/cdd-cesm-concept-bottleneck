# G3 localisation vs radiologist polygons (b2_mammoclip_640x1024, 1225 abnormal images, out-of-fold)

Mean lesion-mask share of the image (random-pointing baseline for the pointing game): 0.094

| explanation | energy | n | pointing game | CAM energy in lesion | IoU (CAM >= 0.5 max) |
|---|---|---|---|---|---|
| black box (true-class logit) | LE | 650 | 0.545 | 0.309 | 0.117 |
| top lesion concept | LE | 650 | 0.600 | 0.242 | 0.125 |
| black box (true-class logit) | DES | 575 | 0.591 | 0.325 | 0.121 |
| top lesion concept | DES | 575 | 0.640 | 0.259 | 0.140 |
| black box (true-class logit) | all | 1225 | 0.567 | 0.317 | 0.119 |
| top lesion concept | all | 1225 | 0.619 | 0.250 | 0.132 |
## Paired tests (same 1,225 images)
- **Pointing game:** concept explanation hit 0.619 vs black box 0.567. Discordant pairs: the concept
  explanation alone hit the lesion on 132 images, the black box alone on 68 (exact McNemar p < 0.0001). The
  random-pointing baseline is 0.094.
- **IoU** (CAM ≥ 0.5 max): concept 0.132 vs black box 0.119 (paired Wilcoxon p = 0.012).
- **Share of CAM energy inside the lesion:** the black box is higher (0.317 vs 0.250, p < 1e-70). The
  concept maps peak on the lesion more often but spread more around it.
- **By label (pointing):**
  - benign: concept 0.416 vs black box 0.275 (the concept explanation helps most here);
  - malignant: 0.797 vs 0.823 (similar).
- **By concept (pointing):**
  - well localised: le_high_density 0.83, des_heterogeneous 0.82, des_irregular 0.76, le_calc_suspicious
    0.73, le_mass 0.66, des_nme 0.64, des_mass_enh 0.64;
  - poorly localised (diffuse findings, as expected): le_asymmetry 0.35, des_focus_foci 0.24.
