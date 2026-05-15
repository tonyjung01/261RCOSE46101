# Retry-Pool Alternative Aggregators — K=3 — 20260515

- retry artifacts: ['20260515_gsm8k_retry_exp_pool_t02_seed42', '20260515_gsm8k_retry_exp_pool_t02_seed43', '20260515_gsm8k_retry_exp_pool_t02_seed44']
- selective manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json`
- flagged subset on test: `60`; full test: `264`

## Flagged-subset accuracy by aggregator

| Aggregator | Correct / flagged | Acc |
|---|---:|---:|
| **base (no retry)** | 18 / 60 | **30.00%** |
| k1 | 24 / 60 | 40.00% |
| majority | 26 / 60 | 43.33% |
| best_by_margin | 21 / 60 | 35.00% |
| best_by_top1 | 21 / 60 | 35.00% |
| confidence_weighted | 22 / 60 | 36.67% |
| two_of_K_else_base | 25 / 60 | 41.67% |
| union_any_correct ← | 35 / 60 | 58.33% |

`union_any_correct` is the oracle ceiling; the others are deployable.

## Full-test deploy (n=264, aggregator on flagged, base elsewhere)

| Aggregator | Full-test acc | delta vs base | fixes | hurts |
|---|---:|---:|---:|---:|
| **P0 baseline (no retry)** | 67.42% | — | 0 | 0 |
| k1 | 70.08% | `+2.65pt` | 14 | 7 |
| majority | 70.45% | `+3.03pt` | 14 | 6 |
| best_by_margin | 68.56% | `+1.14pt` | 12 | 9 |
| best_by_top1 | 68.56% | `+1.14pt` | 12 | 9 |
| confidence_weighted | 68.94% | `+1.52pt` | 12 | 8 |
| two_of_K_else_base | 70.08% | `+2.65pt` | 9 | 2 |

## Mechanism sidecar — samples where union-any correct but majority miss

`9` samples on flagged 60. These are the source of the oracle-vs-majority gap.

| idx | gt | base | K1 ans (corr) | K2 ans (corr) | K3 ans (corr) | majority | best_margin | best_top1 | conf_weighted | 2-of-K-else-base | margins |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 260.0 | 260.0 | None (✗) | 420.0 (✗) | 260.0 (✓) | 420.0 (✗) | 420.0 (✗) | 420.0 (✗) | 420.0 (✗) | 260.0 (✓) | — / 994.6 / 973.7 |
| 43 | 48.0 | 200.0 | 67.0 (✗) | 1.0 (✗) | 48.0 (✓) | 67.0 (✗) | 1.0 (✗) | 1.0 (✗) | 1.0 (✗) | 200.0 (✗) | 273.0 / 666.9 / 172.6 |
| 411 | 1110.0 | 960.0 | 770.0 (✗) | 1670.0 (✗) | 1110.0 (✓) | 770.0 (✗) | 770.0 (✗) | 770.0 (✗) | 770.0 (✗) | 960.0 (✗) | 597.6 / 137.0 / 255.4 |
| 542 | 7.0 | 17.0 | 6.6 (✗) | 8.0 (✗) | 7.0 (✓) | 6.6 (✗) | 6.6 (✗) | 6.6 (✗) | 6.6 (✗) | 17.0 (✗) | 589.5 / 356.8 / 364.2 |
| 692 | 32.0 | 11.0 | 21.0 (✗) | 74.0 (✗) | 32.0 (✓) | 21.0 (✗) | 21.0 (✗) | 21.0 (✗) | 21.0 (✗) | 11.0 (✗) | 443.2 / 38.2 / 212.7 |
| 968 | 227.0 | 177.0 | 237.0 (✗) | 227.0 (✓) | 213.0 (✗) | 237.0 (✗) | 237.0 (✗) | 237.0 (✗) | 237.0 (✗) | 177.0 (✗) | 1342.5 / 467.8 / 146.8 |
| 1130 | 25.0 | 8.0 | 222.0 (✗) | 15.0 (✗) | 25.0 (✓) | 222.0 (✗) | 25.0 (✓) | 25.0 (✓) | 25.0 (✓) | 8.0 (✗) | 317.9 / 389.8 / 833.5 |
| 1182 | 1800.0 | 450.0 | 300.0 (✗) | 380.0 (✗) | 1800.0 (✓) | 300.0 (✗) | 1800.0 (✓) | 1800.0 (✓) | 1800.0 (✓) | 450.0 (✗) | 255.4 / 209.3 / 775.4 |
| 1195 | 320.0 | 320.0 | 330.0 (✗) | 320.0 (✓) | 340.0 (✗) | 330.0 (✗) | 330.0 (✗) | 330.0 (✗) | 330.0 (✗) | 320.0 (✓) | 711.5 / 429.6 / 46.4 |

Reading guide:

- if `best_by_margin` is correct on most of these `(✓)` while `majority` is wrong, then confidence-margin can pull the oracle gain out — best-of-K by confidence works.
- if `2_of_K_else_base` is correct on most because base was correct or because some agreement materializes, the safer policy beats majority by avoiding the wrong 2-of-3 cluster.
- if both alternatives still miss most of these, the diversity is real but the in-vote confidence signal doesn't disambiguate which K-vote to trust.