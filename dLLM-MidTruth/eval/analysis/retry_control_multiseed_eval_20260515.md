# Phase E-Retry-Control-MultiSeed — 20260515

- selective retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260514_gsm8k_retry_exp_testsubset_tau05845_v6/rank_0_generations.json`
- selective manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/retry_subset_gsm8k_test_logistic_broad_tau05845_20260514.json`
- answer kind: `exp_only`
- random-control seeds: `4`

baseline `exp_only` full-test: `67.42%`; P_selective: `70.08%` (`+2.65pt`)

## Per-seed table

| seed | exclude_flagged | overlap | P_random | Δbase | subset base | subset retry | subset Δ | targeting marginal |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 124 | True | 0 | 67.05% | `-0.38pt` | 76.67% | 75.00% | `-1.67pt` | **`+3.03pt`** |
| 125 | True | 0 | 66.29% | `-1.14pt` | 80.00% | 75.00% | `-5.00pt` | **`+3.79pt`** |
| 126 | True | 0 | 67.80% | `+0.38pt` | 78.33% | 80.00% | `+1.67pt` | **`+2.27pt`** |
| 127 | True | 0 | 67.05% | `-0.38pt` | 85.00% | 83.33% | `-1.67pt` | **`+3.03pt`** |

## Aggregate across random seeds

- targeting marginal `(P_selective − P_random)`: mean `+3.03pt` ± `0.62pt`
- random subset-local retry delta: mean `-1.67pt` ± `2.72pt`
- fraction of seeds with positive targeting marginal: `4/4`
- fraction of seeds with negative random-subset retry delta: `3/4`

## Reading guide

- if targeting marginal mean is comfortably positive with a CI that excludes 0, the score-targeting claim survives single-seed noise on this artifact.
- if all seeds have negative random subset-local retry delta, that pattern ("retry on unflagged samples is net-negative") is robust on this artifact, not a single-seed quirk.
- a 95% CI on the mean is approximately `mean ± 1.96 · std / sqrt(K)`; with K=4 random-control seeds the CI is still fairly wide, so this is a robustness check more than a precise estimate.
