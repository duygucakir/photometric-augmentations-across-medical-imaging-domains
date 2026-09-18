# SKIN Binary Experiments Status

### 1. Baselines
| Setup | Status | Folds | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|---|
| baseline_none | ✅ Finished | 5 / 5 | 0.9323 ± 0.0128 | 0.9313 ± 0.0130 | 0.9707 ± 0.0023 |
| baseline_pre | ✅ Finished | 5 / 5 | 0.9271 ± 0.0101 | 0.9259 ± 0.0101 | 0.9669 ± 0.0018 |

### 2. Perturbation Grid (Status Summary)
| Preprocessing | Color | low sigma | med sigma | high sigma |
|---|---|---|---|---|
| **nopre** | R | 🏃 (0 / 5) | ✅ (Acc: 0.925) | ❌ (0/5) |
| **nopre** | G | ❌ (0/5) | ✅ (Acc: 0.911) | ❌ (0/5) |
| **nopre** | B | ✅ (Acc: 0.935) | ✅ (Acc: 0.927) | ✅ (Acc: 0.915) |
| **nopre** | RGB | ❌ (0/5) | ✅ (Acc: 0.890) | ❌ (0/5) |
| **pre** | R | ✅ (Acc: 0.926) | ✅ (Acc: 0.923) | ✅ (Acc: 0.921) |
| **pre** | G | ✅ (Acc: 0.927) | ✅ (Acc: 0.919) | ✅ (Acc: 0.912) |
| **pre** | B | ✅ (Acc: 0.926) | ✅ (Acc: 0.927) | ✅ (Acc: 0.922) |
| **pre** | RGB | ✅ (Acc: 0.926) | ✅ (Acc: 0.921) | ✅ (Acc: 0.868) |


*Legend:*
- ✅ = Finished (all 5 folds completed)
- ❌ = Not started (no folds completed)
- 🏃 = Running (Currently computing folds)
- ⏸️ = Paused/Partial (Some folds completed)

### 3. Detailed Results for Finished Conditions
| Configuration | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| `baseline_none` | 0.9323 ± 0.0128 | 0.9313 ± 0.0130 | 0.9707 ± 0.0023 |
| `baseline_pre` | 0.9271 ± 0.0101 | 0.9259 ± 0.0101 | 0.9669 ± 0.0018 |
| `nopre_B_high` | 0.9146 ± 0.0038 | 0.9131 ± 0.0038 | 0.9617 ± 0.0055 |
| `nopre_B_low` | 0.9352 ± 0.0055 | 0.9341 ± 0.0055 | 0.9724 ± 0.0046 |
| `nopre_B_med` | 0.9274 ± 0.0053 | 0.9262 ± 0.0054 | 0.9718 ± 0.0027 |
| `nopre_G_med` | 0.9107 ± 0.0077 | 0.9095 ± 0.0077 | 0.9618 ± 0.0049 |
| `nopre_RGB_med` | 0.8896 ± 0.0175 | 0.8885 ± 0.0170 | 0.9496 ± 0.0054 |
| `nopre_R_med` | 0.9246 ± 0.0078 | 0.9234 ± 0.0078 | 0.9647 ± 0.0036 |
| `pre_B_high` | 0.9216 ± 0.0076 | 0.9202 ± 0.0077 | 0.9584 ± 0.0067 |
| `pre_B_low` | 0.9261 ± 0.0053 | 0.9249 ± 0.0055 | 0.9699 ± 0.0052 |
| `pre_B_med` | 0.9269 ± 0.0050 | 0.9256 ± 0.0052 | 0.9665 ± 0.0039 |
| `pre_G_high` | 0.9120 ± 0.0061 | 0.9107 ± 0.0063 | 0.9516 ± 0.0043 |
| `pre_G_low` | 0.9267 ± 0.0060 | 0.9254 ± 0.0061 | 0.9703 ± 0.0050 |
| `pre_G_med` | 0.9186 ± 0.0081 | 0.9174 ± 0.0082 | 0.9636 ± 0.0053 |
| `pre_RGB_high` | 0.8683 ± 0.0197 | 0.8665 ± 0.0192 | 0.9292 ± 0.0071 |
| `pre_RGB_low` | 0.9261 ± 0.0087 | 0.9250 ± 0.0089 | 0.9690 ± 0.0040 |
| `pre_RGB_med` | 0.9207 ± 0.0073 | 0.9193 ± 0.0075 | 0.9566 ± 0.0071 |
| `pre_R_high` | 0.9212 ± 0.0083 | 0.9198 ± 0.0085 | 0.9614 ± 0.0072 |
| `pre_R_low` | 0.9256 ± 0.0060 | 0.9243 ± 0.0063 | 0.9705 ± 0.0047 |
| `pre_R_med` | 0.9231 ± 0.0080 | 0.9218 ± 0.0082 | 0.9668 ± 0.0047 |

*Last updated: 2026-05-06 12:54:27*