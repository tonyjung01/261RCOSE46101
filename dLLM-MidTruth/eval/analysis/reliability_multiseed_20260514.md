# E-Diff2 — Multi-seed Outer-Split Robustness — 20260514

Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]; val_frac=0.8; threshold_quantile=0.25

## Per-seed results on GSM8K

| Seed | n_test | exp_only | prob_vote | gated | prob_better | exp_better | fixes | hurts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 264 | 67.42% | 70.08% | 68.94% | 7 | 0 | 4 | 0 |
| 43 | 264 | 72.35% | 73.11% | 73.48% | 5 | 3 | 4 | 1 |
| 44 | 264 | 70.08% | 69.32% | 70.08% | 3 | 5 | 3 | 3 |
| 45 | 264 | 68.18% | 67.80% | 67.80% | 3 | 4 | 1 | 2 |
| 46 | 264 | 67.80% | 65.91% | 68.18% | 4 | 9 | 4 | 3 |
| 47 | 264 | 68.56% | 68.94% | 70.08% | 4 | 3 | 4 | 0 |
| 48 | 264 | 69.32% | 68.94% | 70.08% | 5 | 6 | 4 | 2 |
| 49 | 264 | 72.35% | 69.70% | 71.59% | 4 | 11 | 1 | 3 |
| 50 | 264 | 67.05% | 68.56% | 67.80% | 7 | 3 | 4 | 2 |
| 51 | 264 | 67.42% | 67.05% | 68.56% | 4 | 5 | 3 | 0 |

## Aggregate

- n_seeds = 10
- exp_only test acc:   `69.05% ± 1.96pt`
- prob_vote test acc:  `68.94% ± 1.93pt`
- gated test acc:      `69.66% ± 1.82pt`

### Deltas

- gated − exp_only:  `+0.61pt ± 0.78pt`
- prob_vote − exp_only: `-0.11pt ± 1.55pt`
- gated − prob_vote: `+0.72pt ± 1.11pt`

### Seed-level outcomes

- Fraction of seeds where gated beats exp_only: **70%**
- Fraction of seeds where prob_vote beats exp_only: **40%**
- Fraction of seeds where prob_vote beats gated: **20%**
- Fraction of seeds where gated beats prob_vote: **70%**

- prob_better mean ± std: `4.60 ± 1.43`
- exp_better mean ± std:  `4.90 ± 3.18`

## Notes

- The seed=42 result reported in E4/E6 (`+1.52pt`) corresponds to a single row above; compare it against the mean to see how outlier-like it is.
- If the `gated − prob_vote` mean is consistently negative across seeds, that confirms the differential targetability finding: the threshold rule is dominated by global `prob_vote` on average, and the seed=42 result was a lucky single split.
- If `prob_better − exp_better` is approximately constant across seeds, the test split's `exp_better=0` at seed=42 is the outlier driver.