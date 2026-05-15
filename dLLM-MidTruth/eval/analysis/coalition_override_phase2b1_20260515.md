# Phase 2 B1 — Conservative Answer-Coalition Override — 20260515

Rule: default `exp_only`; override iff some alternative answer has ≥K temporal-readout support, optionally also requiring `logistic_broad_score < tau` and/or `last_valid_answer` agreement. Val-fit, one-shot test eval.

- baseline val acc: `70.62%`
- baseline test acc: `67.42%`
- temporal readouts (5): ['last_valid_answer', 'late_window_majority_q25', 'late_window_majority_q50', 'longest_run_answer', 'most_persistent_answer']

## Val sweep

| K | tau | stable | val_acc | delta | changed | fixes | hurts |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | — | False | 68.44% | `-2.18pt` | 110 | 18 | 41 |
| 2 | — | True | 70.52% | `-0.09pt` | 5 | 1 | 2 |
| 2 | 0.500 | False | 69.95% | `-0.66pt` | 54 | 5 | 12 |
| 2 | 0.500 | True | 70.52% | `-0.09pt` | 5 | 1 | 2 |
| 2 | 0.585 | False | 69.76% | `-0.85pt` | 69 | 9 | 18 |
| 2 | 0.585 | True | 70.52% | `-0.09pt` | 5 | 1 | 2 |
| 3 | — | False | 70.05% | `-0.57pt` | 26 | 2 | 8 |
| 3 | — | True | 70.52% | `-0.09pt` | 1 | 0 | 1 |
| 3 | 0.500 | False | 70.43% | `-0.19pt` | 16 | 1 | 3 |
| 3 | 0.500 | True | 70.52% | `-0.09pt` | 1 | 0 | 1 |
| 3 | 0.585 | False | 70.33% | `-0.28pt` | 21 | 2 | 5 |
| 3 | 0.585 | True | 70.52% | `-0.09pt` | 1 | 0 | 1 |
| 4 | — | False | 70.62% ← | `+0.00pt` | 0 | 0 | 0 |
| 4 | — | True | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 4 | 0.500 | False | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 4 | 0.500 | True | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 4 | 0.585 | False | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 4 | 0.585 | True | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 5 | — | False | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 5 | — | True | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 5 | 0.500 | False | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 5 | 0.500 | True | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 5 | 0.585 | False | 70.62% | `+0.00pt` | 0 | 0 | 0 |
| 5 | 0.585 | True | 70.62% | `+0.00pt` | 0 | 0 | 0 |

**Val-best config**: K=4, tau=None, stable=False, val_acc=`70.62%`

## Test one-shot at val-best config

- test acc: **`67.42%`**
- delta vs exp_only baseline: **`+0.00pt`**
- changed: 0, fixes: 0, hurts: 0
- reason distribution: `{'no_alternative': 231, 'insufficient_coalition': 33}`

## Test diagnostic configs (informational, NOT used for selection)

| K | tau | stable | test_acc | delta | changed | fixes | hurts |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | — | False | 68.94% | `+1.52pt` | 21 | 6 | 2 |
| 3 | — | False | 68.56% | `+1.14pt` | 5 | 3 | 0 |
| 4 | — | False | 67.42% | `+0.00pt` | 0 | 0 | 0 |
| 3 | 0.585 | False | 68.56% | `+1.14pt` | 5 | 3 | 0 |
| 3 | — | True | 67.42% | `+0.00pt` | 0 | 0 | 0 |
| 4 | — | True | 67.42% | `+0.00pt` | 0 | 0 | 0 |

## Decision read

- Test lift `+0.00pt`: ≤ 0. Coalition override does not help. Raw-accuracy line is saturated.