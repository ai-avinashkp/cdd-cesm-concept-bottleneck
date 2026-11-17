# Model comparison (breast level, 566 breasts, patient bootstrap x2000)

| model | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| b1 / black box | 0.655 [0.616, 0.695] | 0.829 [0.803, 0.854] | 0.843 [0.800, 0.880] |
| b1 / concept bottleneck | 0.680 [0.643, 0.718] | 0.858 [0.834, 0.881] | 0.889 [0.855, 0.918] |
| b2_mammoclip_640x1024 / black box | 0.698 [0.660, 0.732] | 0.849 [0.826, 0.871] | 0.866 [0.830, 0.901] |
| b2_mammoclip_640x1024 / concept bottleneck | 0.701 [0.664, 0.738] | 0.867 [0.845, 0.888] | 0.886 [0.854, 0.916] |
| b4_dinov2b_518x770 / black box | 0.636 [0.596, 0.674] | 0.816 [0.791, 0.839] | 0.836 [0.796, 0.873] |
| b4_dinov2b_518x770 / concept bottleneck | 0.654 [0.613, 0.695] | 0.823 [0.796, 0.850] | 0.835 [0.796, 0.871] |

## Paired differences (A - B): estimate [95% CI], P(diff <= 0) under the bootstrap

| A vs B | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| b1 / concept bottleneck vs b1 / black box | +0.025 [-0.009, +0.061], p=0.089 | +0.029 [+0.012, +0.047], p=0.001 | +0.046 [+0.022, +0.072], p=0.000 |
| b2_mammoclip_640x1024 / concept bottleneck vs b2_mammoclip_640x1024 / black box | +0.004 [-0.030, +0.037], p=0.418 | +0.017 [+0.000, +0.035], p=0.022 | +0.020 [+0.000, +0.041], p=0.024 |
| b4_dinov2b_518x770 / concept bottleneck vs b4_dinov2b_518x770 / black box | +0.018 [-0.026, +0.062], p=0.224 | +0.007 [-0.014, +0.028], p=0.240 | -0.001 [-0.022, +0.020], p=0.532 |
| b2_mammoclip_640x1024 / black box vs b1 / black box | +0.042 [+0.007, +0.078], p=0.011 | +0.020 [+0.001, +0.040], p=0.019 | +0.024 [-0.002, +0.051], p=0.034 |
| b2_mammoclip_640x1024 / concept bottleneck vs b1 / concept bottleneck | +0.021 [-0.016, +0.058], p=0.148 | +0.009 [-0.010, +0.028], p=0.168 | -0.002 [-0.021, +0.017], p=0.587 |
| b4_dinov2b_518x770 / black box vs b1 / black box | -0.019 [-0.064, +0.024], p=0.817 | -0.013 [-0.040, +0.013], p=0.825 | -0.007 [-0.041, +0.031], p=0.652 |
| b4_dinov2b_518x770 / concept bottleneck vs b1 / concept bottleneck | -0.027 [-0.074, +0.020], p=0.878 | -0.034 [-0.061, -0.008], p=0.996 | -0.054 [-0.082, -0.025], p=1.000 |

Single training seed per run: the CIs cover test-set (patient) sampling, not training randomness.