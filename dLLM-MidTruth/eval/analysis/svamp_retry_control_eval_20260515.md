# SVAMP Cross-task Retry Control — 20260515

Cross-task transfer of the GSM8K-val-fitted reliability score and `tau=0.5845`. Same score definition, same threshold, no SVAMP-specific tuning.

- n SVAMP samples: `300`
- answer kind on retry: `exp_only`
- selective manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_retry_subset_logistic_broad_score_tau05845_20260515.json`
- random manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_random_retry_subset_complement_seed124_20260515.json`
- selective retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260515_svamp_retry_exp_selective_tau05845/rank_0_generations.json`
- random retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260515_svamp_retry_exp_complement_seed124/rank_0_generations.json`
- selective n: `49`, random n: `49`, overlap: `0`

## Full-SVAMP policies

| Policy | Acc | vs baseline | changed | fixes | hurts |
|---|---:|---:|---:|---:|---:|
| P0 baseline `exp_only` | 86.33% | — | 0 | 0 | 0 |
| P_selective (score-targeted retry) | 86.33% | `+0.00pt` | 9 | 3 | 3 |
| P_random (random-budget retry) | 85.67% | `-0.67pt` | 2 | 0 | 2 |

## Subset-local accuracy (retried slice only)

| Subset | n | base acc | retry acc | delta |
|---|---:|---:|---:|---:|
| selective (score-flagged) | 49 | 55.10% | 55.10% | `+0.00pt` |
| random | 49 | 93.88% | 89.80% | `-4.08pt` |

## Targeting marginal value

`(P_selective − P_random) on full SVAMP = +0.67pt`

Interpretation guide (single seed, single retry artifact per policy, cross-task transfer of the GSM8K-fitted score):

- if `P_selective − P_random > 0` and selective subset-local delta is materially larger than random subset-local delta, the score targeting transfers to SVAMP — score-targeted retry beats random retry on a different task without any SVAMP-specific tuning.
- if `P_selective − P_random ≈ 0` or negative, the score's targeting effect is GSM8K-specific and does not transfer.
- single-seed result; should be read as artifact-level alongside the GSM8K multi-seed control (`+3.03pt ± 0.62pt`).