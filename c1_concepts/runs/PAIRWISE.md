# Pairwise metrics, 3-seed ensemble (breast level, patient-grouped CV; restricted to the two classes)

Columns: AUC; accuracy at 0.5 of the two-class probability; recall of the first class; recall of the second class. 95% CI from 2,000 patient bootstrap resamples.

| pair | model | n | AUC | accuracy | recall (first) | recall (second) |
|---|---|---|---|---|---|---|
| normal vs benign | concept bottleneck | 383 | 0.836 (0.796-0.872) | 0.770 (0.727-0.812) | 0.847 (0.793-0.895) | 0.696 (0.628-0.760) |
| normal vs benign | black box | 383 | 0.792 (0.748-0.837) | 0.718 (0.673-0.762) | 0.878 (0.828-0.927) | 0.562 (0.490-0.633) |
| normal vs benign | AUC difference CBM - black box | | +0.044 (+0.011 to +0.077), p=0.010 | | | |
| normal vs malignant | concept bottleneck | 372 | 0.982 (0.970-0.991) | 0.941 (0.916-0.963) | 0.984 (0.964-1.000) | 0.896 (0.850-0.938) |
| normal vs malignant | black box | 372 | 0.982 (0.970-0.993) | 0.949 (0.926-0.969) | 0.995 (0.983-1.000) | 0.902 (0.857-0.941) |
| normal vs malignant | AUC difference CBM - black box | | -0.001 (-0.007 to +0.007), p=0.820 | | | |
| benign vs malignant | concept bottleneck | 377 | 0.901 (0.871-0.929) | 0.833 (0.795-0.868) | 0.871 (0.824-0.915) | 0.792 (0.734-0.849) |
| benign vs malignant | black box | 377 | 0.883 (0.849-0.914) | 0.836 (0.799-0.871) | 0.830 (0.774-0.882) | 0.842 (0.790-0.890) |
| benign vs malignant | AUC difference CBM - black box | | +0.018 (+0.000 to +0.037), p=0.049 | | | |