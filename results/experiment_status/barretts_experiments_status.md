# HYPERKVASIR Barretts Experiments Status

### 1. Baselines
| Setup | Status | Folds | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|---|
| baseline_none | ✅ Finished | 5 / 5 | 0.9814 ± 0.0019 | 0.7300 ± 0.0330 | 0.8029 ± 0.1095 |
| baseline_pre | ✅ Finished | 5 / 5 | 0.9795 ± 0.0033 | 0.7165 ± 0.0475 | 0.8649 ± 0.0581 |

### 2. Perturbation Grid (Status Summary)
| Preprocessing | Color | low sigma | med sigma | high sigma |
|---|---|---|---|---|
| **nopre** | R | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | G | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | B | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **nopre** | RGB | ❌ (0/5) | ❌ (0/5) | ❌ (0/5) |
| **pre** | R | ✅ (Acc: 0.979) | ✅ (Acc: 0.980) | ✅ (Acc: 0.979) |
| **pre** | G | ✅ (Acc: 0.980) | ✅ (Acc: 0.979) | ✅ (Acc: 0.978) |
| **pre** | B | ✅ (Acc: 0.979) | ✅ (Acc: 0.981) | ✅ (Acc: 0.980) |
| **pre** | RGB | ✅ (Acc: 0.980) | ✅ (Acc: 0.978) | ✅ (Acc: 0.973) |


*Legend:*
- ✅ = Finished (all 5 folds completed)
- ❌ = Not started (no folds completed)
- 🏃 = Running (Currently computing folds)
- ⏸️ = Paused/Partial (Some folds completed)

### 3. Detailed Results for Finished Conditions
| Configuration | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| `baseline_none` | 0.9814 ± 0.0019 | 0.7300 ± 0.0330 | 0.8029 ± 0.1095 |
| `baseline_pre` | 0.9795 ± 0.0033 | 0.7165 ± 0.0475 | 0.8649 ± 0.0581 |
| `pre_B_high` | 0.9798 ± 0.0044 | 0.6668 ± 0.0758 | 0.8326 ± 0.0802 |
| `pre_B_low` | 0.9790 ± 0.0053 | 0.6890 ± 0.0626 | 0.8258 ± 0.0778 |
| `pre_B_med` | 0.9814 ± 0.0036 | 0.7040 ± 0.0727 | 0.8521 ± 0.0640 |
| `pre_G_high` | 0.9778 ± 0.0016 | 0.5451 ± 0.0674 | 0.7741 ± 0.0628 |
| `pre_G_low` | 0.9798 ± 0.0048 | 0.7175 ± 0.0624 | 0.8496 ± 0.0715 |
| `pre_G_med` | 0.9786 ± 0.0038 | 0.6675 ± 0.0473 | 0.8349 ± 0.1181 |
| `pre_RGB_high` | 0.9728 ± 0.0085 | 0.5071 ± 0.0257 | 0.6463 ± 0.0966 |
| `pre_RGB_low` | 0.9805 ± 0.0029 | 0.6847 ± 0.0637 | 0.8538 ± 0.0468 |
| `pre_RGB_med` | 0.9778 ± 0.0026 | 0.5943 ± 0.0563 | 0.8582 ± 0.0725 |
| `pre_R_high` | 0.9786 ± 0.0018 | 0.6314 ± 0.0597 | 0.9045 ± 0.0367 |
| `pre_R_low` | 0.9793 ± 0.0042 | 0.6996 ± 0.0590 | 0.8384 ± 0.0690 |
| `pre_R_med` | 0.9805 ± 0.0036 | 0.6619 ± 0.0818 | 0.8344 ± 0.0537 |

*Last updated: 2026-04-29 14:40:21*