# G2b: binary normal vs tumorous, LE only, Mammo-CLIP, patient-grouped folds (5/5 folds done)

- breast sides evaluated: 566 (normal 189, tumorous 377)
- **AUC 0.869**; accuracy at 0.5: 0.802

| fold | n | AUC | accuracy@0.5 |
|---|---|---|---|
| 0 | 110 | 0.861 | 0.818 |
| 1 | 114 | 0.889 | 0.825 |
| 2 | 112 | 0.898 | 0.804 |
| 3 | 113 | 0.844 | 0.761 |
| 4 | 117 | 0.873 | 0.803 |

Reference: PMC12842785 (same 566-side cohort, LE only, dual-view transformer): AUC 0.992, accuracy 0.969. Our 3-class model (LE+DES) re-expressed as normal vs tumorous: AUC 0.901 (CBM), 0.889 (black box).