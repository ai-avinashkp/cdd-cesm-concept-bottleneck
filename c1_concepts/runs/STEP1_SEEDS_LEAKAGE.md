# Step 1: seeds and leakage (Mammo-CLIP B2, 640x1024)

## A. Training-seed variability (3 seeds complete: [0, 1, 2])

| seed | model | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|---|
| 0 | black box | 0.698 | 0.849 | 0.866 |
| 0 | concept bottleneck | 0.701 | 0.867 | 0.886 |
| 1 | black box | 0.691 | 0.848 | 0.869 |
| 1 | concept bottleneck | 0.735 | 0.871 | 0.894 |
| 2 | black box | 0.701 | 0.853 | 0.888 |
| 2 | concept bottleneck | 0.728 | 0.872 | 0.901 |

| model | accuracy mean ± SD | macro AUC mean ± SD | B-vs-M AUC mean ± SD |
|---|---|---|---|
| black box | 0.697 ± 0.005 | 0.850 ± 0.003 | 0.875 ± 0.012 |
| concept bottleneck | 0.721 ± 0.018 | 0.870 ± 0.003 | 0.894 ± 0.008 |
| CBM − black box | +0.025 ± 0.020 (positive in 3/3 seeds) | +0.020 ± 0.003 (positive in 3/3 seeds) | +0.019 ± 0.006 (positive in 3/3 seeds) |

## B. Leakage: image-level random split vs patient-grouped split (seed 0)

Under the image-level split, on average 99% of each test fold's patients also have images in its training folds.

| split | level | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|---|
| patient-grouped (correct) | image | 0.663 | 0.815 | 0.830 |
| image-level random (leaky) | image | 0.706 | 0.852 | 0.877 |
| **inflation** | image | +0.043 | +0.037 | +0.047 |