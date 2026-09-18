# HYPERKVASIR Esophagitis Experiments Status

### 1. Baselines
| Setup | Status | Folds | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|---|
| baseline_none | ✅ Finished | 5 / 5 | 0.9420 ± 0.0068 | 0.8669 ± 0.0188 | 0.9245 ± 0.0159 |
| baseline_pre | ✅ Finished | 5 / 5 | 0.9376 ± 0.0027 | 0.8631 ± 0.0064 | 0.9417 ± 0.0134 |

### 2. Perturbation Grid (Status Summary)
| Preprocessing | Color | low sigma | med sigma | high sigma |
|---|---|---|---|---|
| **nopre** | R | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | G | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | B | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | RGB | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **pre** | R | ✅ (Acc: 0.937) | ✅ (Acc: 0.928) | ✅ (Acc: 0.913) |
| **pre** | G | ✅ (Acc: 0.938) | ✅ (Acc: 0.907) | ✅ (Acc: 0.899) |
| **pre** | B | ✅ (Acc: 0.938) | ✅ (Acc: 0.936) | ✅ (Acc: 0.920) |
| **pre** | RGB | ✅ (Acc: 0.942) | ✅ (Acc: 0.918) | ✅ (Acc: 0.874) |


*Legend:*
- ✅ = Finished (all 5 folds completed)
- ❌ = Not started (no folds completed)
- 🏃 = Running (Currently computing folds)
- ⏸️ = Paused/Partial (Some folds completed)

### 3. Detailed Results for Finished Conditions
| Configuration | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| `baseline_none` | 0.9420 ± 0.0068 | 0.8669 ± 0.0188 | 0.9245 ± 0.0159 |
| `baseline_pre` | 0.9376 ± 0.0027 | 0.8631 ± 0.0064 | 0.9417 ± 0.0134 |
| `pre_B_high` | 0.9200 ± 0.0098 | 0.8083 ± 0.0287 | 0.9497 ± 0.0092 |
| `pre_B_low` | 0.9376 ± 0.0076 | 0.8639 ± 0.0129 | 0.9344 ± 0.0325 |
| `pre_B_med` | 0.9361 ± 0.0084 | 0.8557 ± 0.0186 | 0.9309 ± 0.0236 |
| `pre_G_high` | 0.8994 ± 0.0196 | 0.6719 ± 0.1137 | 0.8848 ± 0.0198 |
| `pre_G_low` | 0.9382 ± 0.0049 | 0.8652 ± 0.0084 | 0.9425 ± 0.0258 |
| `pre_G_med` | 0.9069 ± 0.0089 | 0.7506 ± 0.0490 | 0.9366 ± 0.0100 |
| `pre_RGB_high` | 0.8736 ± 0.0083 | 0.5401 ± 0.0840 | 0.7408 ± 0.0897 |
| `pre_RGB_low` | 0.9416 ± 0.0047 | 0.8695 ± 0.0107 | 0.9439 ± 0.0134 |
| `pre_RGB_med` | 0.9177 ± 0.0145 | 0.7742 ± 0.0610 | 0.9259 ± 0.0148 |
| `pre_R_high` | 0.9126 ± 0.0154 | 0.7666 ± 0.0630 | 0.9396 ± 0.0154 |
| `pre_R_low` | 0.9367 ± 0.0063 | 0.8611 ± 0.0119 | 0.9363 ± 0.0245 |
| `pre_R_med` | 0.9278 ± 0.0084 | 0.8261 ± 0.0236 | 0.9280 ± 0.0288 |

*Last updated: 2026-05-03 08:57:20*