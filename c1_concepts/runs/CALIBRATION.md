# Calibration of the breast-level outputs (three-seed ensemble, 566 breasts)

Brier score (three classes, lower is better) and top-label ECE (10 bins). 95% CI from 2,000 patient bootstrap resamples.

| output | Brier | ECE |
|---|---|---|
| concept bottleneck | 0.378 (0.345-0.411) | 0.030 (0.020-0.069) |
| black box, fitted | 0.386 (0.352-0.422) | 0.022 (0.019-0.066) |
| black box, averaged | 0.412 (0.377-0.450) | 0.045 (0.033-0.085) |

| A minus B | Brier | ECE |
|---|---|---|
| concept bottleneck minus black box, fitted | -0.008 (-0.035 to +0.017); two-sided p=0.520 | +0.007 (-0.031 to +0.036); two-sided p=0.869 |
| concept bottleneck minus black box, averaged | -0.034 (-0.059 to -0.009); two-sided p=0.005 | -0.016 (-0.050 to +0.023); two-sided p=0.452 |