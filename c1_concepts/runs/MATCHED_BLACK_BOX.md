# Like-for-like black box (three-seed ensemble, 566 breasts, patient-grouped CV)

The averaged black box is the mean of image-label class probabilities. The fitted black box adds a multinomial logistic regression (L2, C=1) on breast-level aggregates of those probabilities, cross-fitted on the same folds as the concept head. 12 inputs: mean and max of each class probability over the LE images and over the recombined images. 95% CI and two-sided p: 2,000 patient bootstrap resamples (paired).

| output | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| concept bottleneck | 0.733 (0.697-0.768) | 0.875 (0.854-0.896) | 0.901 (0.871-0.929) |
| black box, averaged | 0.717 (0.680-0.753) | 0.858 (0.836-0.881) | 0.883 (0.849-0.914) |
| black box, fitted (12 inputs) | 0.719 (0.682-0.755) | 0.871 (0.850-0.892) | 0.895 (0.862-0.924) |
| black box, fitted (6 inputs) | 0.717 (0.681-0.752) | 0.870 (0.849-0.891) | 0.898 (0.866-0.926) |
| black box, fitted (3 inputs) | 0.723 (0.684-0.758) | 0.866 (0.844-0.888) | 0.889 (0.854-0.920) |

## Paired differences

| A minus B | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| concept bottleneck minus black box, fitted (12 inputs) | +0.014 (-0.026 to +0.053); two-sided p=0.472 | +0.004 (-0.012 to +0.021); two-sided p=0.598 | +0.006 (-0.013 to +0.026); two-sided p=0.568 |
| concept bottleneck minus black box, averaged | +0.016 (-0.018 to +0.051); two-sided p=0.388 | +0.017 (+0.000 to +0.034); two-sided p=0.046 | +0.018 (+0.000 to +0.037); two-sided p=0.049 |
| black box, fitted (12 inputs) minus black box, averaged | +0.002 (-0.027 to +0.030); two-sided p=0.982 | +0.013 (+0.001 to +0.024); two-sided p=0.021 | +0.012 (-0.005 to +0.030); two-sided p=0.154 |
| concept bottleneck minus black box, fitted (6 inputs) | +0.016 (-0.020 to +0.052); two-sided p=0.385 | +0.005 (-0.010 to +0.021); two-sided p=0.496 | +0.003 (-0.016 to +0.021); two-sided p=0.777 |
| concept bottleneck minus black box, fitted (3 inputs) | +0.011 (-0.024 to +0.047); two-sided p=0.579 | +0.009 (-0.006 to +0.026); two-sided p=0.266 | +0.012 (-0.006 to +0.032); two-sided p=0.188 |

## Errors by class (argmax)

| output | normal recall | benign recall | malignant recall | benign called normal |
|---|---|---|---|---|
| concept bottleneck | 0.847 | 0.567 | 0.792 | 59 |
| black box, averaged | 0.878 | 0.454 | 0.831 | 81 |
| black box, fitted (12 inputs) | 0.778 | 0.562 | 0.825 | 57 |

## Breasts whose image labels disagree (50) and the others (516): accuracy

| output | disagreeing | others |
|---|---|---|
| concept bottleneck | 0.660 | 0.740 |
| black box, averaged | 0.400 | 0.748 |
| black box, fitted (12 inputs) | 0.520 | 0.738 |

## Malignancy detection (malignant vs normal or benign), score = P(malignant)

| output | AUC | sens @ spec 0.80 | spec @ sens 0.90 |
|---|---|---|---|
| concept bottleneck | 0.935 | 0.913 | 0.843 |
| black box, averaged | 0.933 | 0.913 | 0.817 |
| black box, fitted (12 inputs) | 0.937 | 0.918 | 0.836 |

## Image labels by breast label (images)

| breast label | image Normal | image Benign | image Malignant |
|---|---|---|---|
| Normal | 664 | 0 | 0 |
| Benign | 91 | 587 | 0 |
| Malignant | 2 | 0 | 662 |

## Each seed separately

| seed | output | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|---|
| 0 | concept bottleneck | 0.701 | 0.867 | 0.886 |
| 0 | black box, averaged | 0.698 | 0.849 | 0.866 |
| 0 | black box, fitted (12 inputs) | 0.719 | 0.867 | 0.887 |
| 1 | concept bottleneck | 0.735 | 0.871 | 0.894 |
| 1 | black box, averaged | 0.691 | 0.848 | 0.869 |
| 1 | black box, fitted (12 inputs) | 0.723 | 0.861 | 0.888 |
| 2 | concept bottleneck | 0.728 | 0.872 | 0.901 |
| 2 | black box, averaged | 0.701 | 0.853 | 0.888 |
| 2 | black box, fitted (12 inputs) | 0.700 | 0.864 | 0.897 |

Concept bottleneck minus fitted black box, per seed: accuracy -0.018, +0.012, +0.028 (mean +0.008); macro AUC +0.000, +0.009, +0.008 (mean +0.006); B-vs-M AUC -0.001, +0.005, +0.004 (mean +0.003)

## By machine (means over the three seeds; accuracy / B-vs-M AUC)

| output | machine 1 | machine 2 |
|---|---|---|
| concept bottleneck | 0.725 / 0.898 | 0.657 / 0.762 |
| black box, averaged | 0.700 / 0.880 | 0.636 / 0.729 |
| black box, fitted (12 inputs) | 0.717 / 0.896 | 0.667 / 0.723 |

## Label-only network (no concept loss), seed 0

| output | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| label-only black box, averaged | 0.696 | 0.850 | 0.866 |
| label-only black box, fitted (12 inputs) | 0.705 | 0.861 | 0.882 |

| A minus B | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| seed-0 concept bottleneck minus label-only fitted black box | -0.004 (-0.041 to +0.032); two-sided p=0.908 | +0.006 (-0.012 to +0.025); two-sided p=0.520 | +0.004 (-0.019 to +0.028); two-sided p=0.737 |
| seed-0 fitted black box minus label-only fitted black box | +0.014 (-0.007 to +0.035); two-sided p=0.191 | +0.006 (-0.003 to +0.015); two-sided p=0.189 | +0.005 (-0.003 to +0.016); two-sided p=0.233 |