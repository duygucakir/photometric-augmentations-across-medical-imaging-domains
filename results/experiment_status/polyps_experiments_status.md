# HYPERKVASIR Polyps Experiments Status

### 1. Baselines
| Setup | Status | Folds | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|---|
| baseline_none | ✅ Finished | 5 / 5 | 0.9924 ± 0.0031 | 0.9881 ± 0.0049 | 0.9921 ± 0.0021 |
| baseline_pre | ✅ Finished | 5 / 5 | 0.9934 ± 0.0024 | 0.9896 ± 0.0038 | 0.9953 ± 0.0028 |

### 2. Perturbation Grid (Status Summary)
| Preprocessing | Color | low sigma | med sigma | high sigma |
|---|---|---|---|---|
| **nopre** | R | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | G | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | B | ❌ (0/5) | 🏃 (2 / 5) | ❌ (0/5) |
| **nopre** | RGB | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **pre** | R | ✅ (Acc: 0.993) | ✅ (Acc: 0.990) | ✅ (Acc: 0.983) |
| **pre** | G | ✅ (Acc: 0.989) | ✅ (Acc: 0.985) | ✅ (Acc: 0.972) |
| **pre** | B | ✅ (Acc: 0.993) | ✅ (Acc: 0.990) | ✅ (Acc: 0.984) |
| **pre** | RGB | ✅ (Acc: 0.992) | ✅ (Acc: 0.978) | ✅ (Acc: 0.857) |


*Legend:*
- ✅ = Finished (all 5 folds completed)
- ❌ = Not started (no folds completed)
- 🏃 = Running (Currently computing folds)
- ⏸️ = Paused/Partial (Some folds completed)

### 3. Detailed Results for Finished Conditions
| Configuration | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| `baseline_none` | 0.9924 ± 0.0031 | 0.9881 ± 0.0049 | 0.9921 ± 0.0021 |
| `baseline_pre` | 0.9934 ± 0.0024 | 0.9896 ± 0.0038 | 0.9953 ± 0.0028 |
| `pre_B_high` | 0.9836 ± 0.0066 | 0.9748 ± 0.0097 | 0.9936 ± 0.0036 |
| `pre_B_low` | 0.9926 ± 0.0015 | 0.9884 ± 0.0023 | 0.9926 ± 0.0034 |
| `pre_B_med` | 0.9903 ± 0.0037 | 0.9848 ± 0.0058 | 0.9932 ± 0.0042 |
| `pre_G_high` | 0.9719 ± 0.0062 | 0.9561 ± 0.0101 | 0.9935 ± 0.0036 |
| `pre_G_low` | 0.9891 ± 0.0038 | 0.9831 ± 0.0057 | 0.9952 ± 0.0052 |
| `pre_G_med` | 0.9854 ± 0.0037 | 0.9773 ± 0.0057 | 0.9963 ± 0.0021 |
| `pre_RGB_high` | 0.8574 ± 0.0210 | 0.7165 ± 0.0564 | 0.9218 ± 0.0326 |
| `pre_RGB_low` | 0.9918 ± 0.0026 | 0.9872 ± 0.0040 | 0.9939 ± 0.0014 |
| `pre_RGB_med` | 0.9776 ± 0.0066 | 0.9648 ± 0.0103 | 0.9942 ± 0.0043 |
| `pre_R_high` | 0.9832 ± 0.0044 | 0.9735 ± 0.0071 | 0.9948 ± 0.0020 |
| `pre_R_low` | 0.9932 ± 0.0017 | 0.9893 ± 0.0027 | 0.9929 ± 0.0037 |
| `pre_R_med` | 0.9899 ± 0.0024 | 0.9842 ± 0.0036 | 0.9965 ± 0.0020 |

*Last updated: 2026-04-27 10:24:11*