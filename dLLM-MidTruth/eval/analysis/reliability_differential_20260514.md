# Reliability Score — Differential Targetability Followup — 20260514

Asks whether the reliability score predicts samples where `prob_vote` beats `exp_only` (delta = +1), as opposed to capturing generic difficulty. Pairs with `reliability_retry_20260514.md` and `reliability_retry_policy_20260514.md`.

## Strategy accuracies (global, no gating)

| Split | n | exp_only | prob_vote | union oracle | both correct | prob_better | exp_better |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k_val | 1055 | 70.62% | 69.57% | 71.75% | 68.44% | 12 | 23 |
| gsm8k_test | 264 | 67.42% | 70.08% | 70.08% | 67.42% | 7 | 0 |
| svamp | 300 | 86.33% | 86.67% | 87.00% | 86.00% | 2 | 1 |

**Read**: on a split where `prob_vote` global > `exp_only` global, the right baseline before any reliability gating is the global swap. The reliability score's job is to beat that, not just to beat `exp_only`.

## Score: `combined_score`

### Score vs (delta / y_exp / y_prob) Pearson

| Split | Pearson(score, delta) | Pearson(score, y_exp) | Pearson(score, y_prob) |
|---|---:|---:|---:|
| gsm8k_val | +0.025 | +0.406 | +0.412 |
| gsm8k_test | -0.175 | +0.413 | +0.362 |
| svamp | -0.022 | +0.330 | +0.327 |

**Read**: large |Pearson(score, y_exp)| with small |Pearson(score, delta)| means the score captures generic difficulty, not method-switch signal.

### Quantile breakdown (5 buckets, low score → high score)

#### gsm8k_val

| Q | n | exp_acc | prob_acc | prob_better | tie | exp_better | net swap value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 211 | 40.76% | 40.76% | 10 | 191 | 10 | +0 |
| 1 | 211 | 61.14% | 58.29% | 2 | 201 | 8 | -6 |
| 2 | 211 | 72.04% | 70.14% | 0 | 207 | 4 | -4 |
| 3 | 211 | 86.73% | 86.26% | 0 | 210 | 1 | -1 |
| 4 | 211 | 92.42% | 92.42% | 0 | 211 | 0 | +0 |

#### gsm8k_test

| Q | n | exp_acc | prob_acc | prob_better | tie | exp_better | net swap value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 53 | 37.74% | 45.28% | 4 | 49 | 0 | +4 |
| 1 | 53 | 60.38% | 62.26% | 1 | 52 | 0 | +1 |
| 2 | 52 | 73.08% | 73.08% | 0 | 52 | 0 | +0 |
| 3 | 53 | 77.36% | 81.13% | 2 | 51 | 0 | +2 |
| 4 | 53 | 88.68% | 88.68% | 0 | 53 | 0 | +0 |

#### svamp

| Q | n | exp_acc | prob_acc | prob_better | tie | exp_better | net swap value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 65.00% | 66.67% | 1 | 59 | 0 | +1 |
| 1 | 60 | 93.33% | 91.67% | 0 | 59 | 1 | -1 |
| 2 | 60 | 83.33% | 85.00% | 1 | 59 | 0 | +1 |
| 3 | 60 | 95.00% | 95.00% | 0 | 60 | 0 | +0 |
| 4 | 60 | 95.00% | 95.00% | 0 | 60 | 0 | +0 |

### Net swap (fixes - hurts) at each budget

#### gsm8k_val

| budget | k | score-gated net (acc) | random-gated net (acc) | oracle-gated net (acc) |
|---:|---:|---:|---:|---:|
| 5% | 53 | -1 (70.52%) | -2 (70.43%) | +12 (71.75%) |
| 10% | 106 | -3 (70.33%) | -3 (70.33%) | +12 (71.75%) |
| 15% | 158 | -2 (70.43%) | -3 (70.33%) | +12 (71.75%) |
| 20% | 211 | +0 (70.62%) | -1 (70.52%) | +12 (71.75%) |
| 25% | 264 | -1 (70.52%) | +1 (70.71%) | +12 (71.75%) |
| 30% | 316 | -2 (70.43%) | +3 (70.90%) | +12 (71.75%) |
| 40% | 422 | -6 (70.05%) | +1 (70.71%) | +12 (71.75%) |
| 50% | 528 | -8 (69.86%) | -1 (70.52%) | +12 (71.75%) |

#### gsm8k_test

| budget | k | score-gated net (acc) | random-gated net (acc) | oracle-gated net (acc) |
|---:|---:|---:|---:|---:|
| 5% | 13 | +2 (68.18%) | +0 (67.42%) | +7 (70.08%) |
| 10% | 26 | +3 (68.56%) | +0 (67.42%) | +7 (70.08%) |
| 15% | 40 | +3 (68.56%) | +0 (67.42%) | +7 (70.08%) |
| 20% | 53 | +4 (68.94%) | +0 (67.42%) | +7 (70.08%) |
| 25% | 66 | +4 (68.94%) | +1 (67.80%) | +7 (70.08%) |
| 30% | 79 | +5 (69.32%) | +1 (67.80%) | +7 (70.08%) |
| 40% | 106 | +5 (69.32%) | +2 (68.18%) | +7 (70.08%) |
| 50% | 132 | +5 (69.32%) | +3 (68.56%) | +7 (70.08%) |

#### svamp

| budget | k | score-gated net (acc) | random-gated net (acc) | oracle-gated net (acc) |
|---:|---:|---:|---:|---:|
| 5% | 15 | +0 (86.33%) | +0 (86.33%) | +2 (87.00%) |
| 10% | 30 | +1 (86.67%) | +0 (86.33%) | +2 (87.00%) |
| 15% | 45 | +1 (86.67%) | -1 (86.00%) | +2 (87.00%) |
| 20% | 60 | +1 (86.67%) | -1 (86.00%) | +2 (87.00%) |
| 25% | 75 | +0 (86.33%) | -1 (86.00%) | +2 (87.00%) |
| 30% | 90 | +0 (86.33%) | -1 (86.00%) | +2 (87.00%) |
| 40% | 120 | +0 (86.33%) | -1 (86.00%) | +2 (87.00%) |
| 50% | 150 | +1 (86.67%) | -1 (86.00%) | +2 (87.00%) |

**Read**: score-gated > random-gated means the reliability ranking adds targeting value over a random budget at the same size. score-gated < random-gated means the gating is actively misleading. The oracle column shows the ceiling for that budget.

## Score: `logistic_broad_score`

### Score vs (delta / y_exp / y_prob) Pearson

| Split | Pearson(score, delta) | Pearson(score, y_exp) | Pearson(score, y_prob) |
|---|---:|---:|---:|
| gsm8k_val | -0.003 | +0.452 | +0.446 |
| gsm8k_test | -0.129 | +0.480 | +0.446 |
| svamp | -0.029 | +0.428 | +0.424 |

**Read**: large |Pearson(score, y_exp)| with small |Pearson(score, delta)| means the score captures generic difficulty, not method-switch signal.

### Quantile breakdown (5 buckets, low score → high score)

#### gsm8k_val

| Q | n | exp_acc | prob_acc | prob_better | tie | exp_better | net swap value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 211 | 36.97% | 36.97% | 7 | 197 | 7 | +0 |
| 1 | 211 | 60.19% | 58.77% | 5 | 198 | 8 | -3 |
| 2 | 211 | 75.36% | 72.51% | 0 | 205 | 6 | -6 |
| 3 | 211 | 85.78% | 84.83% | 0 | 209 | 2 | -2 |
| 4 | 211 | 94.79% | 94.79% | 0 | 211 | 0 | +0 |

#### gsm8k_test

| Q | n | exp_acc | prob_acc | prob_better | tie | exp_better | net swap value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 53 | 26.42% | 33.96% | 4 | 49 | 0 | +4 |
| 1 | 53 | 66.04% | 67.92% | 1 | 52 | 0 | +1 |
| 2 | 52 | 71.15% | 73.08% | 1 | 51 | 0 | +1 |
| 3 | 53 | 86.79% | 88.68% | 1 | 52 | 0 | +1 |
| 4 | 53 | 86.79% | 86.79% | 0 | 53 | 0 | +0 |

#### svamp

| Q | n | exp_acc | prob_acc | prob_better | tie | exp_better | net swap value |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 61.67% | 63.33% | 1 | 59 | 0 | +1 |
| 1 | 60 | 90.00% | 88.33% | 0 | 59 | 1 | -1 |
| 2 | 60 | 88.33% | 90.00% | 1 | 59 | 0 | +1 |
| 3 | 60 | 93.33% | 93.33% | 0 | 60 | 0 | +0 |
| 4 | 60 | 98.33% | 98.33% | 0 | 60 | 0 | +0 |

### Net swap (fixes - hurts) at each budget

#### gsm8k_val

| budget | k | score-gated net (acc) | random-gated net (acc) | oracle-gated net (acc) |
|---:|---:|---:|---:|---:|
| 5% | 53 | -1 (70.52%) | -2 (70.43%) | +12 (71.75%) |
| 10% | 106 | -2 (70.43%) | -3 (70.33%) | +12 (71.75%) |
| 15% | 158 | -2 (70.43%) | -3 (70.33%) | +12 (71.75%) |
| 20% | 211 | +0 (70.62%) | -1 (70.52%) | +12 (71.75%) |
| 25% | 264 | +1 (70.71%) | +1 (70.71%) | +12 (71.75%) |
| 30% | 316 | +0 (70.62%) | +3 (70.90%) | +12 (71.75%) |
| 40% | 422 | -3 (70.33%) | +1 (70.71%) | +12 (71.75%) |
| 50% | 528 | -7 (69.95%) | -1 (70.52%) | +12 (71.75%) |

#### gsm8k_test

| budget | k | score-gated net (acc) | random-gated net (acc) | oracle-gated net (acc) |
|---:|---:|---:|---:|---:|
| 5% | 13 | +1 (67.80%) | +0 (67.42%) | +7 (70.08%) |
| 10% | 26 | +2 (68.18%) | +0 (67.42%) | +7 (70.08%) |
| 15% | 40 | +2 (68.18%) | +0 (67.42%) | +7 (70.08%) |
| 20% | 53 | +4 (68.94%) | +0 (67.42%) | +7 (70.08%) |
| 25% | 66 | +4 (68.94%) | +1 (67.80%) | +7 (70.08%) |
| 30% | 79 | +4 (68.94%) | +1 (67.80%) | +7 (70.08%) |
| 40% | 106 | +5 (69.32%) | +2 (68.18%) | +7 (70.08%) |
| 50% | 132 | +5 (69.32%) | +3 (68.56%) | +7 (70.08%) |

#### svamp

| budget | k | score-gated net (acc) | random-gated net (acc) | oracle-gated net (acc) |
|---:|---:|---:|---:|---:|
| 5% | 15 | +0 (86.33%) | +0 (86.33%) | +2 (87.00%) |
| 10% | 30 | +0 (86.33%) | +0 (86.33%) | +2 (87.00%) |
| 15% | 45 | +1 (86.67%) | -1 (86.00%) | +2 (87.00%) |
| 20% | 60 | +1 (86.67%) | -1 (86.00%) | +2 (87.00%) |
| 25% | 75 | +1 (86.67%) | -1 (86.00%) | +2 (87.00%) |
| 30% | 90 | +1 (86.67%) | -1 (86.00%) | +2 (87.00%) |
| 40% | 120 | +0 (86.33%) | -1 (86.00%) | +2 (87.00%) |
| 50% | 150 | +0 (86.33%) | -1 (86.00%) | +2 (87.00%) |

**Read**: score-gated > random-gated means the reliability ranking adds targeting value over a random budget at the same size. score-gated < random-gated means the gating is actively misleading. The oracle column shows the ceiling for that budget.

## Notes

- Net swap at the operational E6 budget (`~25%` for `logistic_broad`) on `gsm8k_test` is the most directly comparable point to the existing E4 / E6 result.
- A clean negative (score-gated ≤ random-gated AND prob_vote global ≥ score-gated swap) would mean E4/E6's published lift comes from prob_vote being better on average rather than from any targeting effect.
- A clean positive (score-gated > random-gated AND ≥ prob_vote global) would justify keeping the operational threshold rule as the recommended policy.
- This pass is offline. No new generation. Same artifacts as Phase E/E2/E4/E5/E6.