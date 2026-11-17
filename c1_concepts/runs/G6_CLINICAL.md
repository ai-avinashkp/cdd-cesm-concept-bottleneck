# C1 sections: clinical operating points (A) and G6 interventions (B); 3-seed Mammo-CLIP ensemble

## A. Malignancy detection (malignant vs normal + benign), breast level

| model | AUC | sens @ spec 0.80 | sens @ spec 0.90 | spec @ sens 0.90 | spec @ sens 0.95 |
|---|---|---|---|---|---|
| concept bottleneck | 0.935 | 0.913 [0.86, 0.95] | 0.836 [0.78, 0.90] | 0.843 [0.74, 0.90] | 0.697 [0.48, 0.80] |
| black box | 0.933 | 0.913 [0.86, 0.95] | 0.869 [0.82, 0.91] | 0.817 [0.70, 0.92] | 0.668 [0.44, 0.80] |

## B. G6 interventions: accuracy (B-vs-M AUC) after correcting k concepts per breast

### sequential head

| policy | k=0 | k=1 | k=2 | k=3 | k=5 | k=8 | k=12 | k=35 |
|---|---|---|---|---|---|---|---|---|
| random | 0.733 (0.901) | 0.734 (0.903) | 0.737 (0.904) | 0.740 (0.905) | 0.748 (0.908) | 0.757 (0.912) | 0.769 (0.918) | 0.777 (0.937) |
| uncertainty | 0.733 (0.901) | 0.761 (0.899) | 0.769 (0.901) | 0.774 (0.896) | 0.770 (0.900) | 0.792 (0.922) | 0.797 (0.931) | 0.777 (0.937) |
| importance | 0.733 (0.901) | 0.749 (0.906) | 0.751 (0.917) | 0.793 (0.918) | 0.795 (0.919) | 0.793 (0.917) | 0.790 (0.921) | 0.777 (0.937) |
| expected impact | 0.733 (0.901) | 0.769 (0.906) | 0.786 (0.905) | 0.788 (0.911) | 0.783 (0.926) | 0.802 (0.931) | 0.799 (0.934) | 0.777 (0.937) |

### independent head

| policy | k=0 | k=1 | k=2 | k=3 | k=5 | k=8 | k=12 | k=35 |
|---|---|---|---|---|---|---|---|---|
| random | 0.687 (0.895) | 0.701 (0.898) | 0.710 (0.901) | 0.720 (0.904) | 0.737 (0.908) | 0.760 (0.914) | 0.795 (0.923) | 0.899 (0.947) |
| uncertainty | 0.687 (0.895) | 0.730 (0.904) | 0.770 (0.905) | 0.792 (0.913) | 0.829 (0.917) | 0.852 (0.928) | 0.864 (0.934) | 0.899 (0.947) |
| importance | 0.687 (0.895) | 0.731 (0.902) | 0.747 (0.906) | 0.767 (0.907) | 0.792 (0.918) | 0.834 (0.928) | 0.862 (0.935) | 0.899 (0.947) |
| expected impact | 0.687 (0.895) | 0.765 (0.913) | 0.814 (0.917) | 0.841 (0.927) | 0.860 (0.934) | 0.871 (0.935) | 0.871 (0.942) | 0.899 (0.947) |

Notes: k=35 = all usable concepts corrected. The independent head is trained on report concepts, so with full correction it reaches the concept ceiling; without correction it may be below the sequential head (train/test input mismatch).