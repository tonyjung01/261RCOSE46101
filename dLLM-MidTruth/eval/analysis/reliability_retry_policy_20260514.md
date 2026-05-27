# Reliability Retry Policy (Phase E6) — 20260514

Operationalized fallback policy: if the reliability score is below a validation-selected threshold `tau`, replace `exp_only` with `prob_vote`.

## Base reference

- GSM8K val base: `70.62%`
- GSM8K test base: `67.42%`
- SVAMP base: `86.33%`
- GSM8K test global `prob_vote`: `70.08%`
- SVAMP global `prob_vote`: `86.67%`

## combined_score — GSM8K validation threshold search

| Quantile | tau | Retry rate | Changed | Acc | Delta | Fixes | Hurts |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5% | -8.7368 | 5.02% | 16 | 70.52% | -0.09% | 2 | 3 |
| 10% | -6.3308 | 10.05% | 31 | 70.33% | -0.28% | 3 | 6 |
| 15% | -5.0042 | 15.07% | 44 | 70.43% | -0.19% | 7 | 9 |
| 20% | -3.9798 | 20.00% | 50 | 70.62% | +0.00% | 10 | 10 |
| 25% | -3.1023 | 25.02% | 55 | 70.52% | -0.09% | 11 | 12 |
| 30% | -2.2367 | 30.05% | 60 | 70.43% | -0.19% | 12 | 14 |
| 40% | -0.4554 | 40.00% | 65 | 70.05% | -0.57% | 12 | 18 |
| 50% | 0.8168 | 50.05% | 67 | 69.86% | -0.76% | 12 | 20 |

### combined_score — chosen threshold

- quantile: `20%`
- tau: `-3.9798`
- val retry rate: `20.00%`
- GSM8K test: `68.56%` (`+1.14%`), changed `13`, fixes `3`, hurts `0`
- SVAMP: `86.33%` (`+0.00%`), changed `4`, fixes `0`, hurts `0`

## logistic_broad_score — GSM8K validation threshold search

| Quantile | tau | Retry rate | Changed | Acc | Delta | Fixes | Hurts |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5% | 0.2935 | 5.02% | 17 | 70.52% | -0.09% | 1 | 2 |
| 10% | 0.3990 | 10.05% | 31 | 70.43% | -0.19% | 3 | 5 |
| 15% | 0.4594 | 15.07% | 37 | 70.43% | -0.19% | 4 | 6 |
| 20% | 0.5304 | 20.00% | 42 | 70.62% | +0.00% | 7 | 7 |
| 25% | 0.5845 | 25.02% | 49 | 70.71% | +0.09% | 9 | 8 |
| 30% | 0.6312 | 30.05% | 55 | 70.62% | +0.00% | 11 | 11 |
| 40% | 0.7072 | 40.00% | 60 | 70.33% | -0.28% | 12 | 15 |
| 50% | 0.7648 | 50.05% | 65 | 69.95% | -0.66% | 12 | 19 |

### logistic_broad_score — chosen threshold

- quantile: `25%`
- tau: `0.5845`
- val retry rate: `25.02%`
- GSM8K test: `68.94%` (`+1.52%`), changed `14`, fixes `4`, hurts `0`
- SVAMP: `86.67%` (`+0.33%`), changed `5`, fixes `1`, hurts `0`

## Current read

- This pass turns the retry budget result into a deployable score threshold.
- If the thresholded policy matches the earlier budget-based result, that suggests the reliability score is stable enough to drive a simple fallback gate.
- It is still an offline substitution test; a true retry policy would need fresh generation latency/cost accounting.
