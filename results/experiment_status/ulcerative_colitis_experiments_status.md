# HYPERKVASIR Ulcerative_colitis Experiments Status

### 1. Baselines
| Setup | Status | Folds | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|---|
| baseline_none | ✅ Finished | 5 / 5 | 0.9841 ± 0.0020 | 0.9719 ± 0.0032 | 0.9923 ± 0.0049 |
| baseline_pre | ✅ Finished | 5 / 5 | 0.9869 ± 0.0037 | 0.9769 ± 0.0066 | 0.9964 ± 0.0020 |

### 2. Perturbation Grid (Status Summary)
| Preprocessing | Color | low sigma | med sigma | high sigma |
|---|---|---|---|---|
| **nopre** | R | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | G | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | B | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | RGB | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **pre** | R | ✅ (Acc: 0.986) | ✅ (Acc: 0.984) | ✅ (Acc: 0.969) |
| **pre** | G | ✅ (Acc: 0.986) | ✅ (Acc: 0.983) | ✅ (Acc: 0.950) |
| **pre** | B | ✅ (Acc: 0.985) | ✅ (Acc: 0.985) | ✅ (Acc: 0.980) |
| **pre** | RGB | ✅ (Acc: 0.986) | ✅ (Acc: 0.977) | ✅ (Acc: 0.904) |


*Legend:*
- ✅ = Finished (all 5 folds completed)
- ❌ = Not started (no folds completed)
- 🏃 = Running (Currently computing folds)
- ⏸️ = Paused/Partial (Some folds completed)

### 3. Detailed Results for Finished Conditions
| Configuration | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| `baseline_none` | 0.9841 ± 0.0020 | 0.9719 ± 0.0032 | 0.9923 ± 0.0049 |
| `baseline_pre` | 0.9869 ± 0.0037 | 0.9769 ± 0.0066 | 0.9964 ± 0.0020 |
| `pre_B_high` | 0.9798 ± 0.0027 | 0.9646 ± 0.0048 | 0.9952 ± 0.0027 |
| `pre_B_low` | 0.9853 ± 0.0033 | 0.9741 ± 0.0056 | 0.9951 ± 0.0032 |
| `pre_B_med` | 0.9855 ± 0.0014 | 0.9743 ± 0.0024 | 0.9960 ± 0.0010 |
| `pre_G_high` | 0.9502 ± 0.0125 | 0.9185 ± 0.0172 | 0.9860 ± 0.0044 |
| `pre_G_low` | 0.9863 ± 0.0026 | 0.9759 ± 0.0044 | 0.9941 ± 0.0020 |
| `pre_G_med` | 0.9828 ± 0.0040 | 0.9702 ± 0.0064 | 0.9952 ± 0.0028 |
| `pre_RGB_high` | 0.9039 ± 0.0142 | 0.7964 ± 0.0391 | 0.9165 ± 0.0330 |
| `pre_RGB_low` | 0.9865 ± 0.0033 | 0.9762 ± 0.0056 | 0.9953 ± 0.0013 |
| `pre_RGB_med` | 0.9774 ± 0.0048 | 0.9600 ± 0.0082 | 0.9927 ± 0.0032 |
| `pre_R_high` | 0.9685 ± 0.0060 | 0.9423 ± 0.0117 | 0.9911 ± 0.0030 |
| `pre_R_low` | 0.9863 ± 0.0016 | 0.9757 ± 0.0032 | 0.9964 ± 0.0026 |
| `pre_R_med` | 0.9839 ± 0.0034 | 0.9711 ± 0.0064 | 0.9927 ± 0.0029 |

*Last updated: 2026-05-04 17:05:20*