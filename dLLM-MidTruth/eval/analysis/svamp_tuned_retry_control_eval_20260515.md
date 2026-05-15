# SVAMP-tuned Retry Control — 20260515

Score refit on SVAMP val (seed=42 outer, 80/20). Evaluated on SVAMP test (n=60).

- selective manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_retry_subset_q25_20260515.json`
- random manifest: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/analysis/svamp_tuned_random_retry_subset_complement_seed124_q25_20260515.json`
- selective retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260515_svamp_retry_exp_svamptuned_q25/rank_0_generations.json`
- random retry artifact: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260515_svamp_retry_exp_svamptuned_complement_seed124/rank_0_generations.json`
- selective n: `14`, random n: `14`, overlap: `0`

## SVAMP test policies (n=60)

| Policy | Acc | vs baseline | changed | fixes | hurts |
|---|---:|---:|---:|---:|---:|
| P0 baseline `exp_only` | 86.67% | — | 0 | 0 | 0 |
| P_selective (SVAMP-tuned, flagged retry) | 88.33% | `+1.67pt` | 1 | 1 | 0 |
| P_random (complement-only) | 86.67% | `+0.00pt` | 0 | 0 | 0 |

## Subset-local accuracy (retried slice only)

| Subset | n | base acc | retry acc | delta |
|---|---:|---:|---:|---:|
| selective (SVAMP-tuned flagged) | 14 | 57.14% | 64.29% | `+7.14pt` |
| random complement | 14 | 100.00% | 100.00% | `+0.00pt` |

## Targeting marginal value

`(P_selective − P_random) on SVAMP test = +1.67pt`

## Notes

- SVAMP test split is small (n=60); flagged budget is `14` (~23%); subset-level magnitudes are noisy at this scale.
- the random complement subset was drawn from the SVAMP-tuned unflagged 46 (overlap with flagged = 0). On this seed it happens to have base acc near 100%, so the random control is essentially a one-sided downside test.
- combine with the offline diagnostics in the selective manifest (e.g. `prob_vote can_fix / base_wrong`) for the mechanism read.