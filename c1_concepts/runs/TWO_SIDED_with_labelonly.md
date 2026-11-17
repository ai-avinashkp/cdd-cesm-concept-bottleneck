# Two-sided paired bootstrap p-values (2,000 patient resamples)

| comparison | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| ensemble CBM vs ensemble black box | +0.016 [-0.018, +0.051] p2=0.388 | +0.017 [+0.000, +0.034] p2=0.046 | +0.018 [+0.000, +0.037] p2=0.049 |
| ensemble CBM vs seed-0 CBM | +0.032 [+0.012, +0.054] p2=0.005 | +0.009 [+0.001, +0.016] p2=0.018 | +0.014 [+0.005, +0.025] p2=0.003 |
| seed-0 CBM vs b6_mammoclip_labelonly black box (label-only) | +0.005 [-0.030, +0.040] p2=0.800 | +0.017 [-0.001, +0.035] p2=0.060 | +0.021 [+0.000, +0.043] p2=0.047 |
| seed-0 multitask black box vs b6_mammoclip_labelonly black box | +0.002 [-0.020, +0.021] p2=0.916 | -0.000 [-0.009, +0.008] p2=0.985 | +0.001 [-0.008, +0.010] p2=0.839 |
| b6_mammoclip_labelonly black box: point metrics | 0.696 | 0.850 | 0.866 |