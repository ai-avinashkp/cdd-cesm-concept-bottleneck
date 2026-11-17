# Breast-label training (check a; seeds [0, 1, 2], ensemble over them; 566 breasts, same folds)

Network as in the paper, but the class loss uses the breast label for every image. Run on one H100 (NITJ HPC); the image-label runs were trained on the laptop GPU. 95% CI and two-sided p: 2,000 patient bootstrap resamples (paired).

| network / output | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| image labels (paper) / concept bottleneck | 0.733 (0.697-0.768) | 0.875 (0.854-0.896) | 0.901 (0.871-0.929) |
| image labels (paper) / black box, fitted | 0.719 (0.682-0.755) | 0.871 (0.850-0.892) | 0.895 (0.862-0.924) |
| image labels (paper) / black box, averaged | 0.717 (0.680-0.753) | 0.858 (0.836-0.881) | 0.883 (0.849-0.914) |
| breast labels / concept bottleneck | 0.707 (0.670-0.741) | 0.865 (0.843-0.887) | 0.894 (0.862-0.923) |
| breast labels / black box, fitted | 0.723 (0.686-0.757) | 0.864 (0.842-0.885) | 0.882 (0.847-0.915) |
| breast labels / black box, averaged | 0.714 (0.677-0.751) | 0.857 (0.834-0.881) | 0.881 (0.848-0.913) |

## Paired differences

| A minus B | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| breast labels / concept bottleneck minus breast labels / black box, fitted | -0.016 (-0.049 to +0.016); two-sided p=0.412 | +0.001 (-0.014 to +0.016); two-sided p=0.873 | +0.012 (-0.007 to +0.032); two-sided p=0.229 |
| breast labels / concept bottleneck minus breast labels / black box, averaged | -0.007 (-0.038 to +0.025); two-sided p=0.706 | +0.008 (-0.009 to +0.024); two-sided p=0.316 | +0.012 (-0.006 to +0.030); two-sided p=0.149 |
| image labels / concept bottleneck (paper) minus breast labels / black box, averaged | +0.019 (-0.014 to +0.054); two-sided p=0.267 | +0.018 (+0.002 to +0.035); two-sided p=0.027 | +0.019 (+0.001 to +0.039); two-sided p=0.042 |
| image labels / concept bottleneck (paper) minus breast labels / black box, fitted | +0.011 (-0.025 to +0.047); two-sided p=0.605 | +0.011 (-0.005 to +0.027); two-sided p=0.160 | +0.019 (-0.001 to +0.040); two-sided p=0.070 |
| breast labels / black box, averaged minus image labels / black box, averaged | -0.004 (-0.030 to +0.023); two-sided p=0.829 | -0.001 (-0.012 to +0.011); two-sided p=0.886 | -0.002 (-0.012 to +0.009); two-sided p=0.760 |
| breast labels / concept bottleneck minus image labels / concept bottleneck (paper) | -0.027 (-0.049 to -0.005); two-sided p=0.015 | -0.010 (-0.018 to -0.003); two-sided p=0.004 | -0.007 (-0.017 to +0.001); two-sided p=0.102 |

## Benign recall and benign breasts called normal

| network / output | benign recall | benign called normal |
|---|---|---|
| image labels / concept bottleneck | 0.567 | 59 |
| image labels / black box, fitted | 0.562 | 57 |
| image labels / black box, averaged | 0.454 | 81 |
| breast labels / concept bottleneck | 0.536 | 61 |
| breast labels / black box, fitted | 0.577 | 55 |
| breast labels / black box, averaged | 0.546 | 60 |

## Each seed of the breast-label network

| seed | output | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|---|
| 0 | concept bottleneck | 0.680 | 0.851 | 0.884 |
| 0 | black box, fitted | 0.708 | 0.856 | 0.872 |
| 0 | black box, averaged | 0.712 | 0.851 | 0.870 |
| 1 | concept bottleneck | 0.721 | 0.870 | 0.891 |
| 1 | black box, fitted | 0.723 | 0.862 | 0.884 |
| 1 | black box, averaged | 0.698 | 0.852 | 0.874 |
| 2 | concept bottleneck | 0.703 | 0.855 | 0.884 |
| 2 | black box, fitted | 0.700 | 0.856 | 0.875 |
| 2 | black box, averaged | 0.700 | 0.848 | 0.880 |

## Per-class recall (three-seed ensembles)

| network / output | normal | benign | malignant |
|---|---|---|---|
| image labels / concept bottleneck | 0.847 | 0.567 | 0.792 |
| image labels / black box, fitted | 0.778 | 0.562 | 0.825 |
| image labels / black box, averaged | 0.878 | 0.454 | 0.831 |
| breast labels / concept bottleneck | 0.794 | 0.536 | 0.798 |
| breast labels / black box, fitted | 0.783 | 0.577 | 0.814 |
| breast labels / black box, averaged | 0.799 | 0.546 | 0.803 |
