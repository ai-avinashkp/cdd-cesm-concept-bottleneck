# Model comparison (breast level, 566 breasts, patient bootstrap x2000)

| model | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| b2_mammoclip_640x1024 / black box | 0.698 [0.660, 0.732] | 0.849 [0.826, 0.871] | 0.866 [0.830, 0.901] |
| b2_mammoclip_640x1024 / concept bottleneck | 0.701 [0.664, 0.738] | 0.867 [0.845, 0.888] | 0.886 [0.854, 0.916] |
| b5_mammoclip_912x1520 / black box | 0.714 [0.678, 0.747] | 0.849 [0.825, 0.872] | 0.856 [0.817, 0.889] |
| b5_mammoclip_912x1520 / concept bottleneck | 0.694 [0.656, 0.730] | 0.859 [0.836, 0.882] | 0.877 [0.842, 0.909] |

## Paired differences (A - B): estimate [95% CI], P(diff <= 0) under the bootstrap

| A vs B | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| b2_mammoclip_640x1024 / concept bottleneck vs b2_mammoclip_640x1024 / black box | +0.004 [-0.030, +0.037], p=0.418 | +0.017 [+0.000, +0.035], p=0.022 | +0.020 [+0.000, +0.041], p=0.024 |
| b5_mammoclip_912x1520 / concept bottleneck vs b5_mammoclip_912x1520 / black box | -0.019 [-0.054, +0.014], p=0.885 | +0.011 [-0.007, +0.029], p=0.128 | +0.022 [-0.001, +0.045], p=0.028 |
| b5_mammoclip_912x1520 / black box vs b2_mammoclip_640x1024 / black box | +0.016 [-0.016, +0.047], p=0.165 | -0.001 [-0.017, +0.015], p=0.551 | -0.011 [-0.033, +0.009], p=0.864 |
| b5_mammoclip_912x1520 / concept bottleneck vs b2_mammoclip_640x1024 / concept bottleneck | -0.007 [-0.040, +0.021], p=0.697 | -0.007 [-0.020, +0.005], p=0.858 | -0.009 [-0.025, +0.007], p=0.862 |

Single training seed per run: the CIs cover test-set (patient) sampling, not training randomness.