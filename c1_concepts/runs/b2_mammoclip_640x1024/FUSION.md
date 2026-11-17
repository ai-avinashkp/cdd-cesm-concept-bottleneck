# Idea A: breast-level 4-view fusion (b2_mammoclip_640x1024, frozen fold-model embeddings, 3 head seeds)

| model | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| attention fusion / black box | 0.686 | 0.848 | 0.880 |
| attention fusion / concept bottleneck | 0.707 | 0.864 | 0.876 |
| mean fusion / black box | 0.675 | 0.837 | 0.870 |
| mean fusion / concept bottleneck | 0.684 | 0.845 | 0.865 |
| per-image averaging (current) / black box | 0.698 | 0.849 | 0.866 |
| per-image averaging (current) / concept bottleneck | 0.701 | 0.867 | 0.886 |

## Paired differences vs per-image averaging (same kind), patient bootstrap x2000

| comparison | accuracy | macro AUC | B-vs-M AUC |
|---|---|---|---|
| attention fusion vs averaging (black box) | -0.012 [-0.046, +0.022] p=0.776 | -0.001 [-0.014, +0.011] p=0.573 | +0.013 [-0.001, +0.028] p=0.035 |
| attention fusion vs averaging (concept bottleneck) | +0.005 [-0.023, +0.034] p=0.369 | -0.003 [-0.017, +0.010] p=0.687 | -0.010 [-0.029, +0.007] p=0.886 |
| mean fusion vs averaging (black box) | -0.023 [-0.054, +0.011] p=0.929 | -0.013 [-0.026, +0.000] p=0.971 | +0.003 [-0.013, +0.019] p=0.368 |
| mean fusion vs averaging (concept bottleneck) | -0.018 [-0.053, +0.016] p=0.856 | -0.022 [-0.040, -0.004] p=0.992 | -0.021 [-0.047, +0.002] p=0.961 |
| attention fusion: CBM vs black box | +0.021 [-0.011, +0.054] p=0.115 | +0.015 [+0.001, +0.030] p=0.021 | -0.004 [-0.025, +0.018] p=0.636 |