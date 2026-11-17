# Seed ensemble of Mammo-CLIP (3 runs), breast level, patient-grouped CV

| model | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| single seed mean / black box | 0.697 | 0.850 | 0.875 |
| single seed mean / concept bottleneck | 0.721 | 0.870 | 0.894 |
| **ensemble (3 seeds) / black box** | 0.717 | 0.858 | 0.883 |
| **ensemble (3 seeds) / concept bottleneck** | 0.733 | 0.875 | 0.901 |

## Paired differences, patient bootstrap x2000

| comparison | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| ensemble CBM vs seed-0 CBM | +0.032 [+0.012, +0.054] p=0.003 | +0.009 [+0.001, +0.016] p=0.009 | +0.014 [+0.005, +0.025] p=0.002 |
| ensemble black box vs seed-0 black box | +0.019 [-0.002, +0.042] p=0.042 | +0.009 [+0.001, +0.017] p=0.011 | +0.016 [+0.006, +0.027] p=0.000 |
| ensemble CBM vs ensemble black box | +0.016 [-0.018, +0.051] p=0.194 | +0.017 [+0.000, +0.034] p=0.023 | +0.018 [+0.000, +0.037] p=0.025 |