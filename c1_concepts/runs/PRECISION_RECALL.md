# Precision, recall and F1, 3-seed ensemble (breast level, patient-grouped CV)

Precision = of the breasts the model called X, the share that really are X. Recall = of the breasts that really are X, the share the model called X. F1 = harmonic mean of the two. 3-class: argmax; pairs: restricted to the two classes, cut-off 0.5 on P(b)/(P(a)+P(b)). 95% CI: 2,000 patient bootstrap resamples.

## 3-class (normal / benign / malignant) (n = 566)

| model | class | precision | recall | F1 |
|---|---|---|---|---|
| concept bottleneck | Normal | 0.693 (0.633-0.748) | 0.847 (0.793-0.895) | 0.762 (0.715-0.803) |
| concept bottleneck | Benign | 0.667 (0.596-0.736) | 0.567 (0.492-0.636) | 0.613 (0.551-0.668) |
| concept bottleneck | Malignant | 0.853 (0.798-0.905) | 0.792 (0.734-0.849) | 0.822 (0.778-0.861) |
| concept bottleneck | **accuracy 0.733 (0.697-0.768); macro F1 0.732 (0.697-0.766)** | | | |
| black box | Normal | 0.643 (0.585-0.698) | 0.878 (0.828-0.927) | 0.743 (0.697-0.784) |
| black box | Benign | 0.677 (0.594-0.761) | 0.454 (0.383-0.526) | 0.543 (0.475-0.610) |
| black box | Malignant | 0.854 (0.800-0.903) | 0.831 (0.778-0.881) | 0.842 (0.801-0.879) |
| black box | **accuracy 0.717 (0.680-0.753); macro F1 0.709 (0.672-0.744)** | | | |

Confusion matrix, concept bottleneck (rows = true, columns = predicted):

| true \ predicted | Normal | Benign | Malignant |
|---|---|---|---|
| Normal | 160 | 29 | 0 |
| Benign | 59 | 110 | 25 |
| Malignant | 12 | 26 | 145 |

Confusion matrix, black box (rows = true, columns = predicted):

| true \ predicted | Normal | Benign | Malignant |
|---|---|---|---|
| Normal | 166 | 22 | 1 |
| Benign | 81 | 88 | 25 |
| Malignant | 11 | 20 | 152 |

## Normal vs Benign (n = 383)

| model | class | precision | recall | F1 |
|---|---|---|---|---|
| concept bottleneck | Normal | 0.731 (0.670-0.788) | 0.847 (0.793-0.895) | 0.784 (0.739-0.824) |
| concept bottleneck | Benign | 0.823 (0.761-0.879) | 0.696 (0.628-0.760) | 0.754 (0.702-0.803) |
| concept bottleneck | **accuracy 0.770 (0.727-0.812); macro F1 0.769 (0.725-0.812)** | | | |
| black box | Normal | 0.661 (0.602-0.715) | 0.878 (0.828-0.927) | 0.755 (0.708-0.797) |
| black box | Benign | 0.826 (0.757-0.892) | 0.562 (0.490-0.633) | 0.669 (0.606-0.728) |
| black box | **accuracy 0.718 (0.673-0.762); macro F1 0.712 (0.665-0.757)** | | | |

Confusion matrix, concept bottleneck (rows = true, columns = predicted):

| true \ predicted | Normal | Benign |
|---|---|---|
| Normal | 160 | 29 |
| Benign | 59 | 135 |

Confusion matrix, black box (rows = true, columns = predicted):

| true \ predicted | Normal | Benign |
|---|---|---|
| Normal | 166 | 23 |
| Benign | 85 | 109 |

## Normal vs Malignant (n = 372)

| model | class | precision | recall | F1 |
|---|---|---|---|---|
| concept bottleneck | Normal | 0.907 (0.869-0.943) | 0.984 (0.964-1.000) | 0.944 (0.920-0.965) |
| concept bottleneck | Malignant | 0.982 (0.960-1.000) | 0.896 (0.850-0.938) | 0.937 (0.909-0.961) |
| concept bottleneck | **accuracy 0.941 (0.916-0.963); macro F1 0.941 (0.915-0.963)** | | | |
| black box | Normal | 0.913 (0.875-0.947) | 0.995 (0.983-1.000) | 0.952 (0.930-0.971) |
| black box | Malignant | 0.994 (0.981-1.000) | 0.902 (0.857-0.941) | 0.946 (0.920-0.968) |
| black box | **accuracy 0.949 (0.926-0.969); macro F1 0.949 (0.926-0.969)** | | | |

Confusion matrix, concept bottleneck (rows = true, columns = predicted):

| true \ predicted | Normal | Malignant |
|---|---|---|
| Normal | 186 | 3 |
| Malignant | 19 | 164 |

Confusion matrix, black box (rows = true, columns = predicted):

| true \ predicted | Normal | Malignant |
|---|---|---|
| Normal | 188 | 1 |
| Malignant | 18 | 165 |

## Benign vs Malignant (n = 377)

| model | class | precision | recall | F1 |
|---|---|---|---|---|
| concept bottleneck | Benign | 0.816 (0.764-0.865) | 0.871 (0.824-0.915) | 0.843 (0.804-0.878) |
| concept bottleneck | Malignant | 0.853 (0.798-0.905) | 0.792 (0.734-0.849) | 0.822 (0.778-0.861) |
| concept bottleneck | **accuracy 0.833 (0.795-0.868); macro F1 0.832 (0.794-0.867)** | | | |
| black box | Benign | 0.847 (0.798-0.893) | 0.830 (0.774-0.882) | 0.839 (0.799-0.874) |
| black box | Malignant | 0.824 (0.770-0.876) | 0.842 (0.790-0.890) | 0.832 (0.792-0.870) |
| black box | **accuracy 0.836 (0.799-0.871); macro F1 0.835 (0.799-0.870)** | | | |

Confusion matrix, concept bottleneck (rows = true, columns = predicted):

| true \ predicted | Benign | Malignant |
|---|---|---|
| Benign | 169 | 25 |
| Malignant | 38 | 145 |

Confusion matrix, black box (rows = true, columns = predicted):

| true \ predicted | Benign | Malignant |
|---|---|---|
| Benign | 161 | 33 |
| Malignant | 29 | 154 |
